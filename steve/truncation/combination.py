from typing import List
from .truncation import Truncation


class Combination(Truncation):
    def __init__(self, truncations: List[Truncation]) -> None:
        self.truncations = truncations
        self._truncated = False

    @property
    def truncated(self) -> bool:
        return self._truncated

    def step(self) -> None:
        truncated = False
        for wrapped_done in self.truncations:
            wrapped_done.step()
            truncated = truncated or wrapped_done.truncated
        self._truncated = truncated

    def reset(self, episode_nr: int = 0) -> None:
        self._truncated = False
        for done in self.truncations:
            done.reset(episode_nr)
