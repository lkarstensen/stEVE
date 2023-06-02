from typing import Optional
from .reward import Reward
from ..intervention import Intervention
from ..interimtarget import InterimTarget, InterimTargetDummy


class TargetReached(Reward):
    def __init__(
        self,
        intervention: Intervention,
        factor: float,
        interim_target: Optional[InterimTarget] = None,
    ) -> None:
        self.intervention = intervention
        self.factor = factor
        self.interim_target = interim_target or InterimTargetDummy()

    def step(self) -> None:
        target_reached = self.intervention.target.reached or self.interim_target.reached
        self.reward = self.factor * target_reached

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
