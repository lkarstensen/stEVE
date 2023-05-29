# pylint: disable=unused-argument
import logging
from typing import List, Optional, Tuple
import numpy as np
import gymnasium as gym
from ...util import EveObject
from ..simulation import Simulation
from ..vesseltree import VesselTree


class Fluoroscopy(EveObject):
    def __init__(
        self,
        simulation: Simulation,
        vessel_tree: VesselTree,
        image_frequency: float = 7.5,
        image_rot_zx: Tuple[float, float] = (0.0, 0.0),
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.simulation = simulation
        self.vessel_tree = vessel_tree
        self.image_rot_zx = image_rot_zx
        self.image_frequency = image_frequency
        vessel_low = vessel_tree.coordinate_space.low
        vessel_high = vessel_tree.coordinate_space.high
        vessel_center = (vessel_high + vessel_low) / 2
        self.image_center = image_center or vessel_center.tolist()
        self.field_of_view = field_of_view

    @property
    def image_space(self) -> gym.Space:
        return gym.spaces.Box(1, 1, (1, 1), dtype=np.uint8)

    @property
    def image(self) -> np.ndarray:
        return np.array([[1]], dtype=np.uint8)

    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        low = self.vessel_cs_to_tracking2d(low)
        high = self.vessel_cs_to_tracking2d(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        coords = self.vessel_tree.centerline_coordinates
        coords = self.vessel_cs_to_tracking2d(coords)
        low = np.min(coords, axis=0)
        low -= 0.1 * np.abs(low)
        high = np.max(coords, axis=0)
        high += 0.1 * np.abs(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        low = self.vessel_cs_to_tracking3d(low)
        high = self.vessel_cs_to_tracking3d(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d_space_episode(self) -> gym.spaces.Box:
        coords = self.vessel_tree.centerline_coordinates
        coords = self.vessel_cs_to_tracking3d(coords)
        low = np.min(coords, axis=0)
        high = np.max(coords, axis=0)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d(self) -> np.ndarray:
        return self.vessel_cs_to_tracking3d(self.simulation.dof_positions)

    @property
    def tracking2d(self) -> np.ndarray:
        return self.tracking3d_to_2d(self.tracking3d)

    @property
    def device_trackings3d(self) -> List[np.ndarray]:
        position = self.tracking3d
        position = np.flip(position)
        point_diff = position[:-1] - position[1:]
        length_btw_points = np.linalg.norm(point_diff, axis=-1)
        cum_length = np.cumsum(length_btw_points)
        inserted_lengths = self.simulation.inserted_lengths

        d_lengths = np.array(inserted_lengths)
        n_devices = d_lengths.size
        n_dofs = cum_length.size
        d_lengths = np.broadcast_to(d_lengths, (n_dofs, n_devices)).transpose()
        cum_length = np.broadcast_to(cum_length, (n_devices, n_dofs))

        diff = np.abs(cum_length - d_lengths)
        idxs = np.argmin(diff, axis=1)

        trackings = [np.flip(position[: idx + 1]) for idx in idxs]
        return trackings

    @property
    def device_trackings2d(self) -> List[np.ndarray]:
        trackings_3d = self.device_trackings3d
        trackings_2d = [self.tracking3d_to_2d(tracking) for tracking in trackings_3d]
        return trackings_2d

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...

    def vessel_cs_to_tracking3d(
        self,
        array: np.ndarray,
    ):
        # negative values as coordinate system is rotated, not array in cs
        rot_matrix = self._get_rot_matrix()
        new_array = np.matmul(rot_matrix, array.T).T

        image_center = np.array(self.image_center)
        image_center_rot_cs = np.matmul(rot_matrix, image_center.T).T
        new_array = new_array - image_center_rot_cs
        if self.field_of_view is not None:
            fov = self.field_of_view
            low = np.array([-fov[0] / 2, -np.inf, -fov[1] / 2])
            high = -low
            low_bound = np.any(new_array < low, axis=-1)
            high_bound = np.any(new_array > high, axis=-1)
            out_of_bounds = low_bound + high_bound
            new_array = new_array[~out_of_bounds]
        return new_array.astype(dtype=np.float32)

    def tracking3d_to_vessel_cs(
        self,
        array: np.ndarray,
    ):
        # negative values as coordinate system is rotated, not array in cs
        rot_matrix = self._get_rot_matrix()

        image_center = np.array(self.image_center)
        image_center_rot_cs = np.matmul(rot_matrix, image_center.T).T
        new_array = array + image_center_rot_cs
        new_array = np.matmul(rot_matrix.T, new_array.T).T
        return new_array.astype(dtype=np.float32)

    def _get_rot_matrix(self):
        rot_z = -self.image_rot_zx[0] * np.pi / 180
        rot_x = -self.image_rot_zx[1] * np.pi / 180
        rotation_matrix_z = np.array(
            [
                [np.cos(rot_z), -np.sin(rot_z), 0],
                [np.sin(rot_z), np.cos(rot_z), 0],
                [0, 0, 1],
            ],
        )

        rotation_matrix_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot_x), -np.sin(rot_x)],
                [0, np.sin(rot_x), np.cos(rot_x)],
            ],
        )
        rotation_matrix = np.matmul(rotation_matrix_z, rotation_matrix_x)
        return rotation_matrix

    def vessel_cs_to_tracking2d(
        self,
        array: np.ndarray,
    ):
        tracking_3d = self.vessel_cs_to_tracking3d(array)
        return self.tracking3d_to_2d(tracking_3d)

    def tracking3d_to_2d(self, tracking3d: np.ndarray):
        return np.delete(tracking3d, 1, -1)
