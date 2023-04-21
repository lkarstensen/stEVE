from typing import Any, List, Optional, Tuple
import logging
import queue
import multiprocessing as mp
import numpy as np
from .device import Device
from .sofacore import SOFACore


def run(
    sofacore: SOFACore,
    task_queue,
    results_queue,
    shutdown_event,
):
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        task_name = task[0]
        args = task[1]
        kwargs = task[2]
        attribute = getattr(sofacore, task_name)
        if callable(attribute):
            results = attribute(*args, **kwargs)
        else:
            results = attribute

        results_queue.put(results)

    sofacore.close()
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


class SOFACoreMP(SOFACore):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        dt_simulation: float = 0.006,
        step_timeout: float = 2,
        restart_n_resets: int = 200,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.dt_simulation = dt_simulation
        self.step_timeout = step_timeout
        self.restart_n_resets = restart_n_resets

        self._devices = None

        self._reset_count = 0

        self.simulation_error = False

        self.camera = None
        self.root = None

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

    def add_devices(self, devices: List[Device]):
        super().add_devices(devices)
        self._last_inserted_lengths = [0.0] * len(self._devices)
        self._last_rotations = [0.0] * len(self._devices)

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

    def do_sofa_steps(self, action: np.ndarray, n_steps):
        if self._task_queue is not None:
            self._task_queue.put(["do_sofa_steps", [action, n_steps], {}])
            self._get_result(timeout=self.step_timeout)

    def reset_sofa_devices(self):
        if self._task_queue is not None:
            self._task_queue.put(["reset_sofa_devices", [], {}])
            self._get_result(timeout=self.step_timeout)

    def reset(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        add_visual: bool = False,
        display_size: Optional[Tuple[int, int]] = None,
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
        target_size: Optional[float] = None,
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
                    [insertion_point, insertion_direction, mesh_path, False],
                    {},
                ]
            )

            self._get_result(timeout=30)
            self.simulation_error = False
            self._insertion_point = insertion_point
            self._insertion_direction = insertion_direction
            self._mesh_path = mesh_path
        self._reset_count += 1

    def _new_sofa_process(self):
        self.logger.debug("Starting new sofa process")
        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        process_sofa_core = SOFACore(self.dt_simulation)
        process_sofa_core.add_devices(self._devices)
        self._sofa_process = mp.Process(
            target=run,
            args=[
                process_sofa_core,
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
