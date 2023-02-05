from .reward import Reward
from ..target import Target


class TargetReached(Reward):
    def __init__(self, target: Target, factor: float) -> None:
        self.target = target
        self.factor = factor

    def step(self) -> None:
        self.reward = self.factor * self.target.reached

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
