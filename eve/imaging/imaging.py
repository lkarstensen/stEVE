from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from ..util import EveObject


class Imaging(EveObject, ABC):
    @property
    @abstractmethod
    def image_space(self) -> gym.Space:
        ...

    @property
    @abstractmethod
    def x_ray_image(self) -> np.ndarray:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...
