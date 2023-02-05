from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

from ..util import EveObject


class Observation(EveObject, ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.obs: np.ndarray = None

    @property
    @abstractmethod
    def space(self) -> gym.spaces.Box:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...

    def __call__(self) -> np.ndarray:
        return self.obs
