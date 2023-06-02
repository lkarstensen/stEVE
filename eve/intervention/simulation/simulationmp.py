from typing import Any, Dict, List, Optional, Tuple
import logging
import queue
import multiprocessing as mp
import numpy as np
from ..device import Device
from .simulation import Simulation


def run(
    simu_dict: Dict,
    task_queue: queue.Queue,
    results_queue: queue.Queue,
    shutdown_event,
):
    simulation = Simulation.from_config_dict(simu_dict)
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        task_name = task[0]
        args = task[1]
        kwargs = task[2]
        attribute = getattr(simulation, task_name)
        if callable(attribute):
            results = attribute(*args, **kwargs)
        else:
            results = attribute

        results_queue.put(results)

    simulation.close()
    while True:
        try:
            results_queue.get(timeout=0.1)
        except queue.Empty:
            results_queue.close()
            break
    while True:
        try:
            task_queue.get(timeout=0.1)
        except queue.Empty:
            task_queue.close()
            break


class SimulationMP(Simulation):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        simulation: Simulation,
        step_timeout: float = 2,
        restart_n_resets: int = 200,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.simulation = simulation
        self.step_timeout = step_timeout
        self.restart_n_resets = restart_n_resets

        self._reset_count = 0

        self.simulation_error = False

        self._sofa_process: mp.Process = None
        self._task_queue: mp.Queue = None
        self._result_queue: mp.Queue = None
        self._shutdown_event: mp.Event = None
        self._last_dof_positions = np.array([[0.0, 0.0, 0.0]])
        self._last_inserted_lengths = None
        self._last_rotations = None

        self._insertion_point = np.empty(())
        self._insertion_direction = np.empty(())
        self._mesh_path: str = None

    @property
    def dof_positions(self) -> np.ndarray:
        if self._task_queue is None:
            return self._last_dof_positions

        self._task_queue.put(["dof_positions", (), {}])
        dof_positions = self._get_result(
            timeout=self.step_timeout, default_value=self._last_dof_positions
        )
        self._last_dof_positions = dof_positions
        return dof_positions

    @property
    def inserted_lengths(self) -> List[float]:
        if self._task_queue is None:
            return self._last_inserted_lengths

        self._task_queue.put(["inserted_lengths", (), {}])
        inserted_lengths = self._get_result(
            timeout=self.step_timeout, default_value=self._last_inserted_lengths
        )
        self._last_inserted_lengths = inserted_lengths
        return inserted_lengths

    @property
    def rotations(self) -> List[float]:
        if self._task_queue is None:
            return self._last_rotations

        self._task_queue.put(["rotations", (), {}])
        rotations = self._get_result(
            timeout=self.step_timeout, default_value=self._last_rotations
        )
        self._last_rotations = rotations
        return rotations

    def close(self):
        self._close_sofa_process()

    def do_steps(self, action: np.ndarray, duration):
        if self._task_queue is not None:
            self._task_queue.put(["do_steps", [action, duration], {}])
            self._get_result(timeout=self.step_timeout)

    def reset_devices(self):
        if self._task_queue is not None:
            self._task_queue.put(["reset_devices", [], {}])
            self._get_result(timeout=self.step_timeout)

    def reset(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        devices: List[Device],
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
        vessel_visual_path: Optional[str] = None,
        seed: int = None,
    ):
        if self._sofa_process is None:
            self._new_sofa_process()
        elif self._reset_count > 0 and self._reset_count % self.restart_n_resets == 0:
            self._close_sofa_process()
            self._new_sofa_process()

        if (
            np.any(insertion_point != self._insertion_point)
            or np.any(insertion_direction != self._insertion_direction)
            or mesh_path != self._mesh_path
        ):
            self._task_queue.put(
                [
                    "reset",
                    [
                        insertion_point,
                        insertion_direction,
                        mesh_path,
                        devices,
                        None,
                        None,
                        None,
                        seed,
                    ],
                    {},
                ]
            )

            self._get_result(timeout=30)
            self.simulation_error = False
            self._insertion_point = insertion_point
            self._insertion_direction = insertion_direction
            self._mesh_path = mesh_path
        self._reset_count += 1

    def add_interim_targets(self, positions: List[Tuple[float, float, float]]):
        raise RuntimeError("This Class can't be used with visualization stuff")

    def remove_interim_target(self, interim_target):
        raise RuntimeError("This Class can't be used with visualization stuff")

    def _new_sofa_process(self):
        simu_dict = self.simulation.get_config_dict()
        self.logger.debug("Starting new sofa process")
        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._sofa_process = mp.Process(
            target=run,
            args=[
                simu_dict,
                self._task_queue,
                self._result_queue,
                self._shutdown_event,
            ],
        )
        self._sofa_process.start()
        self._reset_count = 0
        self._insertion_point = np.empty(())
        self._insertion_direction = np.empty(())
        self._mesh_path = None

    def _get_result(self, timeout: float, default_value: Any = -1):
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            self.logger.warning("Killing sofa because of timeout when getting results")
            self._kill_sofa_process()
            self.simulation_error = True
            return default_value

    def _kill_sofa_process(self):
        self._shutdown_event.set()
        self._sofa_process.kill()
        self._sofa_process.join()
        self._cleanup_sofa_process()

    def _close_sofa_process(self):
        self._shutdown_event.set()
        self._sofa_process.join(5)
        if self._sofa_process.exitcode is None:
            self._sofa_process.kill()
            self._sofa_process.join()
        self._cleanup_sofa_process()

    def _cleanup_sofa_process(self):
        self._sofa_process.close()
        self._sofa_process = None
        self._task_queue.close()
        self._task_queue = None
        self._result_queue.close()
        self._result_queue = None
