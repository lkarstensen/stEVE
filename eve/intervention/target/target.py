from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np

from ...util import EveObject
from ...util.coordtransform import tracking3d_to_2d
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
        return tracking3d_to_2d(self.coordinates3d)

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "coordinates3d": self.coordinates3d,
            "coordinates2d": self.coordinates2d,
            "reached": self.reached,
            "threshold": self.threshold,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "reached": self.reached,
        }
        return deepcopy(state)
