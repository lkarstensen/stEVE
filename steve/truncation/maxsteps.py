from .truncation import Truncation

class MaxSteps(Truncation):
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self._step_counter = 0

    @property
    def truncated(self) -> bool:
        return self._step_counter >= self.max_steps

    def step(self) -> None:
        self._step_counter += 1

    def reset(self, episode_nr: int = 0) -> None:
        self._step_counter = 0
