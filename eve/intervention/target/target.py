from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ...util import EveObject
from ..fluoroscopy import Fluoroscopy


class Target(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset().
    # Coordinates are in the tracking coordinate space
    coordinates3d: np.ndarray
    reached: bool
    threshold: float
    fluoroscopy: Fluoroscopy

    @property
    def coordinates2d(self) -> np.ndarray:
        return self.fluoroscopy.tracking3d_to_2d(self.coordinates3d)

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...
