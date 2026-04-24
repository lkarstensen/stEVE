from typing import Optional
import numpy as np

from .reward import Reward
from ..intervention import Intervention
from ..interimtarget import InterimTarget


class TipToTargetDistDelta(Reward):
    def __init__(
        self,
        factor: float,
        intervention: Intervention,
        interim_target: Optional[InterimTarget],
    ) -> None:
        self.factor = factor
        self.intervention = intervention
        self.interim_target = interim_target
        self._last_dist = None
        self._last_target = None

    def step(self) -> None:
        tip = self.intervention.fluoroscopy.tracking3d[0]
        if self.interim_target.coordinates3d is not None:
            target = self.interim_target.coordinates3d
        else:
            target = self.intervention.target.coordinates3d

        dist = np.linalg.norm(tip - target)

        dist_delta = dist - self._last_dist
        self._last_dist = dist
        if np.all(target == self._last_target):
            self.reward = -dist_delta * self.factor
        else:
            self.reward = 0.0
            self._last_target = target

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        tip = self.intervention.fluoroscopy.tracking3d[0]
        target = self.intervention.target.coordinates3d
        dist = np.linalg.norm(tip - target)
        self._last_dist = dist
        self._last_target = target
