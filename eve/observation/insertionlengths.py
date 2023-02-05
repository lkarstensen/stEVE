import numpy as np

from .observation import Observation, gym
from ..intervention import Intervention


class InsertionLengths(Observation):
    def __init__(
        self, intervention: Intervention, name: str = "inserted_lengths"
    ) -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def space(self) -> gym.spaces.Box:
        high = np.array(
            self.intervention.device_lengths_maximum.values(), dtype=np.float32
        )
        low = np.zeros_like(high)
        return gym.spaces.Box(high=high, low=low, dtype=np.float32)

    def step(self) -> None:
        self.obs = np.array(
            self.intervention.device_lengths_inserted.values(), dtype=np.float32
        )

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
