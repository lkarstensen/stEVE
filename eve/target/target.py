from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import gymnasium as gym
from ..intervention.intervention import Intervention

from ..util import EveObject


class Target(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset().
    # Coordinates are in the tracking coordinate space as 3d coordinates
    coordinates_vessel_cs: np.ndarray
    reached: bool
    threshold: float
    intervention: Intervention

    @property
    def coordinates3d(self) -> np.ndarray:
        return self.intervention.vessel_cs_to_tracking3d(self.coordinates_vessel_cs)

    @property
    def coordinates2d(self) -> np.ndarray:
        return self.intervention.tracking3d_to_2d(self.coordinates3d)

    @property
    def coordinate_space2d(self) -> gym.spaces.Box:
        return self.intervention.tracking2d_space

    @property
    def coordinate_space3d(self) -> gym.spaces.Box:
        return self.intervention.tracking3d_space

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    def step(self) -> None:
        position = self.intervention.tracking3d[0]
        position_to_target = self.coordinates3d - position
        if np.linalg.norm(position_to_target) < self.threshold:
            self.reached = True
        else:
            self.reached = False
