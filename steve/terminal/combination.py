from . import Terminal
from typing import List


class Combination(Terminal):
    def __init__(self, terminals: List[Terminal]) -> None:
        self.terminals = terminals
        self._terminal = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    def step(self) -> None:
        terminal = False
        for wrapped_done in self.terminals:
            wrapped_done.step()
            terminal = terminal or wrapped_done.terminal
        self._terminal = terminal

    def reset(self, episode_nr: int = 0) -> None:
        self._terminal = False
        for done in self.terminals:
            done.reset(episode_nr)
