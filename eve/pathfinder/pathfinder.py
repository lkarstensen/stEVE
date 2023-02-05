from abc import ABC, abstractmethod
from typing import List
import numpy as np
import gymnasium as gym

from ..vesseltree import BranchingPoint
from ..util import EveObject


class Pathfinder(EveObject, ABC):
    @property
    @abstractmethod
    def path_length(self) -> float:
        ...

    @property
    @abstractmethod
    def path_points(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def path_branching_points(self) -> List[BranchingPoint]:
        ...

    @property
    @abstractmethod
    def coordinate_space(self) -> gym.spaces.Box:
        ...

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self, episode_nr=0) -> None:
        pass
