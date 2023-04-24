from typing import Dict, List, Optional
import logging
import numpy as np
import gymnasium as gym

from .intervention import Intervention

from ..vesseltree import VesselTree
from .device import Device
from .sofacore import SOFACore
from .sofacoremp import SOFACoreMP


class Simulation(Intervention):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        lao_rao_deg: float = 0.0,
        cra_cau_deg: float = 0.0,
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
        sofacore_mp: bool = False,
        mp_timeout_step: float = 2,
        mp_restart_n_resets: int = 200,
        sofa_core: Optional[SOFACore] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.vessel_tree = vessel_tree
        self.devices = devices
        self.lao_rao_deg = lao_rao_deg
        self.cra_cau_deg = cra_cau_deg
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.image_frequency = image_frequency
        self.dt_simulation = dt_simulation
        self.sofacore_mp = sofacore_mp
        self.mp_timeout_step = mp_timeout_step
        self.mp_restart_n_resets = mp_restart_n_resets
        self.sofa_core = sofa_core

        velocity_limits = tuple(device.velocity_limit for device in devices)
        self.velocity_limits = np.array(velocity_limits, dtype=np.float32)
        self.last_action = np.zeros_like(self.velocity_limits, dtype=np.float32)

        self.init_visual_nodes = False
        self.display_size = (1, 1)
        self.target_size = 1
        if sofa_core is not None:
            self._sofa_core = sofa_core
        elif sofacore_mp:
            self._sofa_core = SOFACoreMP(
                dt_simulation,
                mp_timeout_step,
                mp_restart_n_resets,
            )
        else:
            self._sofa_core = SOFACore(dt_simulation)

        self._sofa_core.add_devices(self.devices)

    @property
    def simulation_error(self) -> bool:
        return self._sofa_core.simulation_error

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
        return self._sofa_core.dof_positions

    @property
    def tracking2d(self) -> np.ndarray:
        return self.tracking3d_to_2d(self.tracking3d)

    @property
    def tracking3d(self) -> np.ndarray:
        tracking = self.instrument_position_vessel_cs
        tracking = self.vessel_cs_to_tracking3d(tracking)
        return tracking

    @property
    def device_lengths_inserted(self) -> Dict[Device, float]:
        lengths = self._sofa_core.inserted_lengths
        return dict(zip(self.devices, lengths))

    @property
    def device_lengths_maximum(self) -> Dict[Device, float]:
        return {device: device.length for device in self.devices}

    @property
    def device_rotations(self) -> Dict[Device, float]:
        rots = self._sofa_core.rotations
        return dict(zip(self.devices, rots))

    @property
    def device_diameters(self) -> Dict[Device, float]:
        return {device: device.radius * 2 for device in self.devices}

    @property
    def sofa_camera(self):
        return self._sofa_core.camera

    @property
    def sofa_root(self):
        return self._sofa_core.root

    @property
    def sofa_target_node(self):
        return self._sofa_core.target_node

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.action_space.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        inserted_lengths = np.array(self._sofa_core.inserted_lengths)
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

        self._sofa_core.do_sofa_steps(
            action, int((1 / self.image_frequency) / self.dt_simulation)
        )

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        # pylint: disable=unused-argument

        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction
        self._sofa_core.reset(
            insertion_point=ip_pos,
            insertion_direction=ip_dir,
            mesh_path=self.vessel_tree.mesh_path,
            add_visual=self.init_visual_nodes,
            display_size=self.display_size,
            coords_low=self.vessel_tree.coordinate_space.low,
            coords_high=self.vessel_tree.coordinate_space.high,
            target_size=self.target_size,
            vessel_visual_path=self.vessel_tree.visu_mesh_path,
        )

    def reset_devices(self) -> None:
        self._sofa_core.reset_sofa_devices()

    def reload_sofa(self):
        # self._sofa_core.reload_sofa()
        self.reset()

    def close(self):
        self._sofa_core.close()

    def vessel_cs_to_tracking2d(
        self,
        array: np.ndarray,
    ):
        rotated_array = self.vessel_cs_to_tracking3d(array)
        rotated_array = np.delete(rotated_array, 1, axis=-1)
        return rotated_array

    def vessel_cs_to_tracking3d(
        self,
        array: np.ndarray,
    ):
        lao_rao_rad = self.lao_rao_deg * np.pi / 180
        cra_cau_rad = self.cra_cau_deg * np.pi / 180

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
        # transpose such that matrix multiplication works
        rotated_array = np.matmul(array, rotation_matrix)
        return rotated_array
