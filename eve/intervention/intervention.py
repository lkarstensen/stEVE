from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

from .device import Device
from ..util import EveObject


class Intervention(EveObject, ABC):
    devices: List[Device]
    instrument_position_vessel_cs: np.ndarray
    device_lengths_inserted: Dict[Device, float]
    device_rotations: Dict[Device, float]
    last_action: np.ndarray
    tracking2d_space: gym.spaces.Box
    tracking2d_space_episode: gym.spaces.Box
    tracking3d_space: gym.spaces.Box
    tracking3d_space_episode: gym.spaces.Box

    @property
    def tracking3d(self) -> np.ndarray:
        return self.vessel_cs_to_tracking3d(self.instrument_position_vessel_cs)

    @property
    def tracking2d(self) -> np.ndarray:
        return self.tracking3d_to_2d(self.tracking3d)

    @property
    def device_trackings3d(self) -> Dict[Device, np.ndarray]:
        position = self.tracking3d
        position = np.flip(position)
        point_diff = position[:-1] - position[1:]
        length_btw_points = np.linalg.norm(point_diff, axis=-1)
        cum_length = np.cumsum(length_btw_points)
        inserted_lengths = self.device_lengths_inserted

        d_lengths = np.array(list(inserted_lengths.values()))
        n_devices = d_lengths.size
        n_dofs = cum_length.size
        d_lengths = np.broadcast_to(d_lengths, (n_dofs, n_devices)).transpose()
        cum_length = np.broadcast_to(cum_length, (n_devices, n_dofs))

        diff = np.abs(cum_length - d_lengths)
        idxs = np.argmin(diff, axis=1)

        trackings = [np.flip(position[: idx + 1]) for idx in idxs]
        device_trackings = {}
        for tracking, device in zip(trackings, inserted_lengths.keys()):
            device_trackings[device] = tracking
        return device_trackings

    @property
    def device_trackings2d(self) -> Dict[Device, np.ndarray]:
        res_3d = self.device_trackings3d
        res2d = {
            device: np.delete(tracking, 1, -1) for device, tracking in res_3d.items()
        }
        return res2d

    @property
    def device_lengths_maximum(self) -> Dict[Device, float]:
        return {device: device.length for device in self.devices}

    @property
    def device_diameters(self) -> Dict[Device, float]:
        return {device: device.radius * 2 for device in self.devices}

    @property
    def action_space(self) -> gym.spaces.Box:
        velocity_limits = np.array([device.velocity_limit for device in self.devices])
        low = -velocity_limits
        high = velocity_limits
        return gym.spaces.Box(low=low, high=high)

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        ...

    @abstractmethod
    def reset_devices(self) -> None:
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def vessel_cs_to_tracking3d(self, array: np.ndarray):
        ...

    def vessel_cs_to_tracking2d(self, array: np.ndarray):
        tracking3d = self.vessel_cs_to_tracking3d(array)
        return self.tracking3d_to_2d(tracking3d)

    def tracking3d_to_2d(self, tracking3d: np.ndarray):
        return np.delete(tracking3d, 1, -1)
