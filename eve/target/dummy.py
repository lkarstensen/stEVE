from typing import Optional
import numpy as np
from ..intervention.intervention import Intervention

from .target import Target


class TargetDummy(Target):
    def __init__(self, intervention: Intervention, threshold: float) -> None:
        self.threshold = threshold
        self.intervention = intervention
        self.coordinates_vessel_cs = np.zeros((3,), dtype=np.float32)
        self.reached = False
        self._coordinates3d = None
        self._coordinates2d = None

    @property
    def coordinates2d(self) -> np.ndarray:
        if self._coordinates2d is None:
            return super().coordinates2d
        return self._coordinates2d

    @coordinates2d.setter
    def coordinates2d(self, coordinates2d: np.ndarray) -> np.ndarray:
        self._coordinates2d = coordinates2d
        self.coordinates3d = np.insert(coordinates2d, 1, 0.0, -1)

    @property
    def coordinates3d(self) -> np.ndarray:
        if self._coordinates3d is None:
            return super().coordinates3d
        return self._coordinates3d

    @coordinates3d.setter
    def coordinates3d(self, coordinates3d: np.ndarray) -> np.ndarray:
        self._coordinates3d = coordinates3d
        self.coordinates_vessel_cs = coordinates3d

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False
