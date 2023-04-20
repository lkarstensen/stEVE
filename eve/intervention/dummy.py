from typing import List
import numpy as np
import gymnasium as gym

from .device import Device
from .intervention import Intervention


class InterventionDummy(Intervention):
    def __init__(
        self,
        devices: List[Device],
        tracking_low: np.ndarray,
        tracking_high: np.ndarray,
    ) -> None:
        self.devices = devices
        self.tracking_low = np.array(tracking_low)
        self.tracking_high = np.array(tracking_high)
        self.instrument_position_vessel_cs = np.zeros((2, 3))
        self.device_lengths_inserted = {device: 0.0 for device in devices}
        self.device_rotations = {device: 0.0 for device in devices}
        self.last_action = np.zeros((len(self.devices), 2), dtype=np.float32)

    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        low = self.tracking_low
        low = low if low.shape[-1] == 2 else np.delete(low, 1, -1)
        high = self.tracking_high
        high = high if high.shape[-1] == 2 else np.delete(high, 1, -1)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        return self.tracking2d_space

    @property
    def tracking3d_space(self) -> gym.spaces.Box:
        low = self.tracking_low
        low = low if low.shape[-1] == 3 else np.insert(low, 1, -1.0, axis=-1)
        high = self.tracking_high
        high = high if high.shape[-1] == 3 else np.insert(high, 1, 1.0, axis=-1)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d_space_episode(self) -> gym.spaces.Box:
        return self.tracking3d_space

    def step(self, action: np.ndarray) -> None:
        self.last_action = action

    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        self.last_action *= 0.0

    def reset_devices(self) -> None:
        ...

    def close(self):
        ...

    def vessel_cs_to_tracking2d(self, array: np.ndarray):
        return np.delete(array, 1, -1)

    def vessel_cs_to_tracking3d(self, array: np.ndarray):
        return array
