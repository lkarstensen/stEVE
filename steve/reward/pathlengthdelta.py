from .reward import Reward
from ..pathfinder import Pathfinder


class PathLengthDelta(Reward):
    def __init__(self, pathfinder: Pathfinder, factor: float) -> None:
        self.pathfinder = pathfinder
        self.factor = factor
        self._last_path_length = None

    def step(self) -> None:
        path_length = self.pathfinder.path_length
        path_length_delta = path_length - self._last_path_length
        self.reward = -path_length_delta * self.factor
        self._last_path_length = path_length

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._last_path_length = self.pathfinder.path_length
