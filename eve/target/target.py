from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import gymnasium as gym
from ..intervention.intervention import Intervention

from ..util import EveObject


class Target(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset():
    coordinates: np.ndarray
    potential_targets: Optional[np.ndarray]

    def __init__(self, intervention: Intervention, threshold: float) -> None:
        self.intervention = intervention
        self.threshold = threshold

        self.reached: bool = False

    @property
    @abstractmethod
    def coordinate_space(self) -> gym.spaces.Box:
        ...

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False

    def step(self) -> None:
        position = self.intervention.instrument_position_vessel_cs[0]
        position_to_target = self.coordinates - position
        if np.linalg.norm(position_to_target) < self.threshold:
            self.reached = True
        else:
            self.reached = False
