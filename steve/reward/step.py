from .reward import Reward


class Step(Reward):
    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.reward = factor

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
