from . import Terminal
from ..intervention import Intervention


class TargetReached(Terminal):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    @property
    def terminal(self) -> bool:
        return self.intervention.target.reached

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
