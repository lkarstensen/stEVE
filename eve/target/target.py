from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from ..intervention import Intervention

from ..util import EveObject


class Target(EveObject, ABC):
    def __init__(self, intervention: Intervention, threshold: float) -> None:
        self.intervention = intervention
        self.threshold = threshold

        self.coordinates: np.ndarray = None
        self.reached: bool = False

    @property
    @abstractmethod
    def coordinate_space(self) -> gym.spaces.Box:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...

    def step(self) -> None:
        position = self.intervention.tracking_ground_truth[0]
        position_to_target = self.coordinates - position
        if np.linalg.norm(position_to_target) < self.threshold:
            self.reached = True
        else:
            self.reached = False
