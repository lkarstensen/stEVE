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
        final_only_after_all_interim: bool = True,
    ) -> None:
        self.intervention = intervention
        self.factor = factor
        self.interim_target = interim_target or InterimTargetDummy()
        self.final_only_after_all_interim = final_only_after_all_interim

    def step(self) -> None:
        if self.final_only_after_all_interim:
            if self.interim_target.coordinates3d is None:
                target_reached = self.intervention.target.reached
            else:
                target_reached = self.interim_target.reached
        else:
            target_reached = (
                self.intervention.target.reached or self.interim_target.reached
            )
        self.reward = self.factor * target_reached

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
