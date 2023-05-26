from .observation import Observation, gym
from math import sin, cos
import numpy as np

from ..intervention.intervention import Intervention


class Rotations(Observation):
    def __init__(self, intervention: Intervention, name: str = "rotation") -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def space(self) -> gym.spaces.Box:
        n_rotations = len(self.intervention.device_rotations)
        shape = (n_rotations, 2)
        low = -np.ones(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        rotation_data = self.intervention.device_rotations
        state = [[sin(rotation), cos(rotation)] for rotation in rotation_data]
        self.obs = np.array(state, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
