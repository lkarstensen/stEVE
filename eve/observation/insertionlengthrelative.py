import numpy as np

from ..intervention import Intervention, Device
from .observation import Observation, gym


class InsertionLengthRelative(Observation):
    def __init__(
        self,
        intervention: Intervention,
        device: Device,
        relative_to_device: Device,
        name: str = None,
    ) -> None:
        name = name or f"{device.name}_length_relative_to_{relative_to_device.name}"
        super().__init__(name)
        self.intervention = intervention
        self.device = device
        self.relative_to_device = relative_to_device

    @property
    def space(self) -> gym.spaces.Box:
        high = self.intervention.device_lengths_maximum[self.device]
        high = np.array(high, dtype=np.float32)
        low = -self.intervention.device_lengths_maximum[self.relative_to_device]
        low = np.array(low, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        inserted_lengths = self.intervention.device_lengths_inserted
        relative_length = (
            inserted_lengths[self.device] - inserted_lengths[self.relative_to_device]
        )
        self.obs = np.array(relative_length, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
