from abc import ABC, abstractmethod
import numpy as np

from ..intervention import Intervention, List, Tuple
from ..vesseltree import VesselTree


class Simulation3D(Intervention, ABC):
    def __init__(
        self,
        vessel_tree: VesselTree,
        image_frequency: float,
        dt_simulation: float,
        velocity_limits: List[Tuple[float, float]],
    ) -> None:
        super().__init__(vessel_tree, image_frequency, dt_simulation, velocity_limits)
        self.initialized_last_reset = True
        self.root = None
        self.sofa_initialized_2 = False

    @abstractmethod
    def _unload_sofa(
        self,
    ):
        ...

    @abstractmethod
    def _do_sofa_step(self, action: np.ndarray):
        ...

    @abstractmethod
    def _reset_sofa_devices(self):
        ...

    @abstractmethod
    def _init_sofa(
        self,
        insertion_point: np.ndarray,
        insertion_direction: np.ndarray,
        mesh_path: str,
    ):
        ...

    @staticmethod
    def _calculate_insertion_pose(
        insertion_point: np.ndarray, insertion_direction: np.ndarray
    ):

        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        original_direction = np.array([1.0, 0.0, 0.0])
        if np.all(insertion_direction == original_direction):
            w0 = 1.0
            xyz0 = [0.0, 0.0, 0.0]
        elif np.all(np.cross(insertion_direction, original_direction) == 0):
            w0 = 0.0
            xyz0 = [0.0, 1.0, 0.0]
        else:
            half = (original_direction + insertion_direction) / np.linalg.norm(
                original_direction + insertion_direction
            )
            w0 = np.dot(original_direction, half)
            xyz0 = np.cross(original_direction, half)
        xyz0 = list(xyz0)
        pose = list(insertion_point) + list(xyz0) + [w0]
        return pose
