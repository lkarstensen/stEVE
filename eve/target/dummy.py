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

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False
