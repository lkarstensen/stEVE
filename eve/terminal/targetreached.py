from . import Terminal
from ..target import Target


class TargetReached(Terminal):
    def __init__(self, target: Target) -> None:
        self.target = target

    @property
    def terminal(self) -> bool:
        return self.target.reached

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
