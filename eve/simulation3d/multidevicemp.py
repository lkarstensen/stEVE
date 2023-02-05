from copy import deepcopy
from typing import Any, List
from multiprocessing.synchronize import Event
import queue
import logging
import multiprocessing as mp
import numpy as np

from .multidevice import MultiDevice, Device3D, VesselTree
from .guidewiremp import run


class MutliDeviceMP(MultiDevice):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device3D],
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
        sofa_native_gui: bool = False,
        step_timeout: float = 10,
        sofa_restart_episodes: float = 25,
    ) -> None:
        super().__init__(
            vessel_tree,
            devices,
            stop_device_at_tree_end,
            image_frequency,
            dt_simulation,
            sofa_native_gui,
        )
        self.step_timeout = step_timeout
        self.sofa_restart_episodes = sofa_restart_episodes

        self._sofa_process: mp.Process = None
        self._task_queue: mp.Queue = None
        self._result_queue: mp.Queue = None
        self._shutdown_event: Event = None
        self.logger = logging.getLogger(self.__module__)

    @property
    def tracking_ground_truth(self) -> np.ndarray:
        default_return = np.array([0.0])
        if self._task_queue is not None:
            self._task_queue.put(["tracking_ground_truth", (), {}])
            return self._get_result(default_return)
        return default_return

    @property
    def device_lengths_inserted(self) -> List[float]:
        default_return = [0.0] * len(self.devices)
        if self._task_queue is not None:
            self._task_queue.put(["device_lengths_inserted", (), {}])
            return self._get_result(default_return)
        return default_return

    @property
    def device_rotations(self) -> List[float]:
        default_return = [0.0] * len(self.devices)
        if self._task_queue is not None:
            self._task_queue.put(["device_rotations", (), {}])
            return self._get_result(default_return)
        return default_return

    @property
    def tracking_per_device(self) -> List[np.ndarray]:
        default_return = np.array([[0, 0]] * len(self.devices))
        if self._task_queue is not None:
            self._task_queue.put(["tracking_per_device", (), {}])
            return self._get_result(default_return)
        return default_return

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        if self._sofa_process is None:
            self._new_sofa_process()
        elif episode_nr > 0 and episode_nr % self.sofa_restart_episodes == 0:
            log_msg = f"Restarting SOFA process at episode nr. {episode_nr}."
            self.logger.info(log_msg)
            self._close_sofa_process()
            self._new_sofa_process()

        super().reset(episode_nr)
        self.simulation_error = False

    def close(self):
        if self._sofa_process is not None:
            self._close_sofa_process()

    def _unload_sofa(self):
        self._task_queue.put(["_unload_sofa", (), {}])
        self._get_result()

    def _do_sofa_step(self, action):
        self._task_queue.put(["_do_sofa_step", [action], {}])
        self._get_result()

    def _reset_sofa_devices(self):
        self._task_queue.put(["_reset_sofa_devices", (), {}])
        self._get_result()

    def _init_sofa(self, insertion_point, insertion_direction, mesh_path):
        self._task_queue.put(
            ["_init_sofa", [insertion_point, insertion_direction, mesh_path], {}]
        )
        self._get_result()

    def _get_result(self, default_value: Any = -1):
        try:
            return self._result_queue.get(timeout=self.step_timeout)
        except queue.Empty:
            self.logger.warning(
                "restarting sofa because of timeout when getting results"
            )
            self._kill_sofa_process()
            self._new_sofa_process()
            self.reset(-1)
            self.simulation_error = True
            return default_value

    def _new_sofa_process(self):
        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._sofa_process = mp.Process(
            target=run,
            args=[
                deepcopy(
                    MultiDevice(
                        self.vessel_tree,
                        self.devices,
                        self.stop_device_at_tree_end,
                        self.image_frequency,
                        self.dt_simulation,
                        self.sofa_native_gui,
                    )
                ),
                self._task_queue,
                self._result_queue,
                self._shutdown_event,
            ],
        )
        self._sofa_process.start()

    def _kill_sofa_process(self):
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
        self._loaded_mesh = None
        self._sofa_initialized = False
