from typing import List, Optional, Tuple
import logging
import numpy as np
import gymnasium as gym

from .simulation import Simulation

from ..vesseltree import VesselTree
from .device import Device
from .sofacore import SOFACore


class MonoPlaneStatic(Simulation):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        sofa_core: SOFACore,
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        image_rot_zx: Tuple[float, float] = (0.0, 0.0),
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.vessel_tree = vessel_tree
        self.devices = devices
        self.image_rot_zx = image_rot_zx
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.image_frequency = image_frequency
        self.sofa_core = sofa_core

        vessel_low = self.vessel_tree.coordinate_space.low
        vessel_high = self.vessel_tree.coordinate_space.high
        vessel_center = (vessel_high + vessel_low) / 2
        self.image_center = image_center or vessel_center.tolist()

        self.field_of_view = field_of_view

        velocity_limits = tuple(device.velocity_limit for device in devices)
        self.velocity_limits = np.array(velocity_limits, dtype=np.float32)
        self.last_action = np.zeros_like(self.velocity_limits, dtype=np.float32)

        self.sofa_core.add_devices(self.devices)

    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        low = self.vessel_cs_to_tracking2d(low)
        high = self.vessel_cs_to_tracking2d(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space_episode.low
        high = self.vessel_tree.coordinate_space_episode.high
        low = self.vessel_cs_to_tracking2d(low)
        high = self.vessel_cs_to_tracking2d(high)
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
        low = self.vessel_tree.coordinate_space_episode.low
        high = self.vessel_tree.coordinate_space_episode.high
        low = self.vessel_cs_to_tracking3d(low)
        high = self.vessel_cs_to_tracking3d(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def instrument_position_vessel_cs(self) -> np.ndarray:
        return self.sofa_core.dof_positions

    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.sofa_core.inserted_lengths

    @property
    def device_rotations(self) -> List[float]:
        return self.sofa_core.rotations

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.action_space.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        inserted_lengths = np.array(self.sofa_core.inserted_lengths)
        max_lengths = np.array([device.length for device in self.devices])

        mask = np.where(inserted_lengths + action[:, 0] / self.image_frequency <= 0.0)
        action[mask, 0] = 0.0
        mask = np.where(
            inserted_lengths + action[:, 0] / self.image_frequency >= max_lengths
        )
        action[mask, 0] = 0.0
        tip = self.instrument_position_vessel_cs[0]
        if self.stop_device_at_tree_end and self.vessel_tree.at_tree_end(tip):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = action[:, 0] / self.image_frequency
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.last_action = action

        self.sofa_core.do_sofa_steps(action, (1 / self.image_frequency))

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction
        self.sofa_core.reset(
            insertion_point=ip_pos,
            insertion_direction=ip_dir,
            mesh_path=self.vessel_tree.mesh_path,
            coords_low=self.vessel_tree.coordinate_space.low,
            coords_high=self.vessel_tree.coordinate_space.high,
            vessel_visual_path=self.vessel_tree.visu_mesh_path,
        )

    def reset_devices(self) -> None:
        self.sofa_core.reset_sofa_devices()

    def close(self):
        self.sofa_core.close()

    def vessel_cs_to_tracking2d(
        self,
        array: np.ndarray,
    ):
        tracking_3d = self.vessel_cs_to_tracking3d(array)
        return self.tracking3d_to_2d(tracking_3d)

    def vessel_cs_to_tracking3d(
        self,
        array: np.ndarray,
    ):
        # negative values as coordinate system is rotated, not array in cs
        lao_rao_rad = -self.image_rot_zx[0] * np.pi / 180
        cra_cau_rad = -self.image_rot_zx[1] * np.pi / 180

        rotation_matrix_lao_rao = np.array(
            [
                [np.cos(lao_rao_rad), -np.sin(lao_rao_rad), 0],
                [np.sin(lao_rao_rad), np.cos(lao_rao_rad), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        rotation_matrix_cra_cau = np.array(
            [
                [1, 0, 0],
                [0, np.cos(cra_cau_rad), -np.sin(cra_cau_rad)],
                [0, np.sin(cra_cau_rad), np.cos(cra_cau_rad)],
            ],
            dtype=np.float32,
        )
        rotation_matrix = np.matmul(rotation_matrix_lao_rao, rotation_matrix_cra_cau)
        # calc array in new
        new_array = np.matmul(rotation_matrix, array.T).T

        image_center = np.array(self.image_center)
        image_center_rot_cs = np.matmul(rotation_matrix, image_center.T).T
        new_array = new_array - image_center_rot_cs
        if self.field_of_view is not None:
            fov = self.field_of_view
            low = np.array([-fov[0] / 2, -np.inf, -fov[1] / 2])
            high = -low
            low_bound = np.any(new_array < low, axis=-1)
            high_bound = np.any(new_array > high, axis=-1)
            out_of_bounds = low_bound + high_bound
            new_array = new_array[~out_of_bounds]

        return new_array
