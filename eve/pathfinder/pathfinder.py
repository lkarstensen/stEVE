from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from ..util import EveObject
from ..vesseltree import VesselTree
from ..intervention import Intervention
from ..target import Target


class Pathfinder(EveObject, ABC):
    path_length: float
    path_points3d: np.ndarray
    path_branching_points3d: np.ndarray
    vesseltree: VesselTree
    intervention: Intervention
    target: Target

    @property
    def path_points2d(self) -> np.ndarray:
        return self.intervention.tracking3d_to_2d(self.path_points3d)

    @property
    def path_branching_points2d(self) -> np.ndarray:
        return self.intervention.tracking3d_to_2d(self.path_branching_points3d)

    @property
    def coordinate_space2d(self) -> gym.spaces.Box:
        return self.intervention.tracking2d_space

    @property
    def coordinate_space3d(self) -> gym.spaces.Box:
        return self.intervention.tracking3d_space

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self, episode_nr=0) -> None:
        pass
