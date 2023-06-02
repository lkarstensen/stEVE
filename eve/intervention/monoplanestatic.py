# pylint: disable=unused-argument
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from .intervention import Intervention
from .target import Target
from .vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy
from .device import Device
from .simulation import Simulation


class MonoPlaneStatic(Intervention):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        simulation: Simulation,
        fluoroscopy: Fluoroscopy,
        target: Target,
        stop_device_at_tree_end: bool = True,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.devices = devices
        self.target = target
        self.fluoroscopy = fluoroscopy
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.simulation = simulation
        self._np_random = np.random.default_rng()

        self.velocity_limits = np.array(
            [device.velocity_limit for device in self.devices]
        )
        self.last_action = np.zeros_like(self.velocity_limits)

    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.simulation.inserted_lengths

    @property
    def device_rotations(self) -> List[float]:
        return self.simulation.rotations

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.length for device in self.devices]

    @property
    def device_diameters(self) -> List[float]:
        return [device.sofa_device.radius * 2 for device in self.devices]

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-self.velocity_limits, high=self.velocity_limits)

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.velocity_limits.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        self.last_action = action

        inserted_lengths = np.array(self.device_lengths_inserted)
        max_lengths = np.array(self.device_lengths_maximum)
        duration = 1 / self.fluoroscopy.image_frequency
        mask = np.where(inserted_lengths + action[:, 0] * duration <= 0.0)
        action[mask, 0] = 0.0
        mask = np.where(inserted_lengths + action[:, 0] * duration >= max_lengths)
        action[mask, 0] = 0.0
        tip = self.simulation.dof_positions[0]
        if self.stop_device_at_tree_end and self.vessel_tree.at_tree_end(tip):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = action[:, 0] * duration
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.vessel_tree.step()
        self.simulation.do_steps(action, duration)
        self.fluoroscopy.step()
        self.target.step()

    def reset(
        self,
        episode_number: int = 0,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(episode_number, vessel_seed)
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction
        self.simulation.reset(
            insertion_point=ip_pos,
            insertion_direction=ip_dir,
            mesh_path=self.vessel_tree.mesh_path,
            devices=self.devices,
            coords_low=self.vessel_tree.coordinate_space.low,
            coords_high=self.vessel_tree.coordinate_space.high,
            vessel_visual_path=self.vessel_tree.visu_mesh_path,
        )
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(episode_number, target_seed)
        self.fluoroscopy.reset(episode_number)

    def close(self) -> None:
        self.simulation.close()

    def reset_devices(self) -> None:
        self.simulation.reset_devices()
