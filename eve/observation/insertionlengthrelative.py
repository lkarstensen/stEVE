import numpy as np

from ..intervention import Intervention
from .observation import Observation, gym


class InsertionLengthRelative(Observation):
    def __init__(
        self,
        intervention: Intervention,
        device_idx: int,
        relative_to_device_idx: int,
        name: str = None,
    ) -> None:
        name = (
            name or f"Device_length_{device_idx}_relative_to_{relative_to_device_idx}"
        )
        super().__init__(name)
        self.intervention = intervention
        self.device_idx = device_idx
        self.relative_to_device_idx = relative_to_device_idx

    @property
    def space(self) -> gym.spaces.Box:
        high = self.intervention.device_lengths_maximum[self.device_idx]
        high = np.array(high, dtype=np.float32)
        low = -self.intervention.device_lengths_maximum[self.relative_to_device_idx]
        low = np.array(low, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        inserted_lengths = self.intervention.device_lengths_inserted
        relative_length = (
            inserted_lengths[self.device_idx]
            - inserted_lengths[self.relative_to_device_idx]
        )
        self.obs = np.array(relative_length, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
