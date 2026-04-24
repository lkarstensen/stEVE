from typing import Dict, Any
from .info import Info


class Steps(Info):
    def __init__(self, name: str = "steps") -> None:
        super().__init__(name)
        self.steps = 0

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.steps}

    def step(self) -> None:
        self.steps += 1

    def reset(self, episode_nr: int = 0) -> None:
        self.steps = 0
