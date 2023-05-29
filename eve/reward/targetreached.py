from .reward import Reward
from ..intervention import Intervention


class TargetReached(Reward):
    def __init__(self, intervention: Intervention, factor: float) -> None:
        self.intervention = intervention
        self.factor = factor

    def step(self) -> None:
        self.reward = self.factor * self.intervention.target.reached

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
