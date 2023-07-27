from typing import Optional
import numpy as np
from ..fluoroscopy import Fluoroscopy

from .target import Target


class TargetDummy(Target):
    def __init__(self, fluoroscopy: Fluoroscopy, threshold: float) -> None:
        self.fluoroscopy = fluoroscopy
        self.threshold = threshold
        self.coordinates2d = None
        self.coordinates3d = None
        self.reached = False

    def step(self) -> None:
        if self.coordinates3d is not None:
            position_to_target = self.coordinates3d - self.fluoroscopy.tracking3d[0]
        elif self.coordinates2d is not None:
            position_to_target = self.coordinates2d - self.fluoroscopy.tracking2d[0]
        else:
            raise ValueError(
                "either 2D or 3D must be given for position and target coordinates"
            )
        self.reached = (
            True if np.linalg.norm(position_to_target) < self.threshold else False
        )

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False
