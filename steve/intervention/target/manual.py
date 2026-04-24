import random
from typing import List, Optional, Tuple
import numpy as np
from ..fluoroscopy import Fluoroscopy

from .target import Target
from ...util.coordtransform import vessel_cs_to_tracking3d


class Manual(Target):
    def __init__(
        self,
        targets_vessel_cs: List[Tuple[float, float, float]],
        threshold: float,
        fluoroscopy: Fluoroscopy,
    ) -> None:
        self.targets_vessel_cs = targets_vessel_cs
        self.threshold = threshold
        self.fluoroscopy = fluoroscopy

        self.reached = False
        self.coordinates3d = np.zeros((3,), dtype=np.float32)

        self._rng = random.Random()

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False
        if seed is not None:
            self._rng = random.Random(seed)
        target_vessel_cs = self._rng.choice(self.targets_vessel_cs)
        self.coordinates3d = vessel_cs_to_tracking3d(
            target_vessel_cs,
            self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center,
            self.fluoroscopy.field_of_view,
        )
