import numpy as np

from .reward import Reward
from ..intervention.intervention import Intervention
from ..target import Target


class TipToTargetDistDelta(Reward):
    def __init__(
        self, intervention: Intervention, target: Target, factor: float
    ) -> None:
        self.intervention = intervention
        self.target = target
        self.factor = factor
        self._last_dist = None
        self._last_target = None

    def step(self) -> None:
        tip = self.intervention.instrument_position_vessel_cs[0]
        target = self.target.coordinates2d
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
        tip = self.intervention.instrument_position_vessel_cs[0]
        target = self.target.coordinates2d
        dist = np.linalg.norm(tip - target)
        self._last_dist = dist
        self._last_target = target
