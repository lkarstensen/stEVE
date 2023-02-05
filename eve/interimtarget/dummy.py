import numpy as np
import gymnasium as gym

from .interimtarget import InterimTarget


class Dummy(InterimTarget):
    def __init__(self) -> None:
        super().__init__(None, 0)
        self.coordinates = np.array([0, 0, 0])
        self.all_coordinates = np.array([[0, 0, 0]])
        self.reached = False

    @property
    def coordinate_space(self):
        return gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([0, 0, 0]))

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
