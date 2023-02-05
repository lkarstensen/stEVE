from .reward import Reward
from ..pathfinder import Pathfinder


class PathLength(Reward):
    def __init__(self, pathfinder: Pathfinder, factor: float) -> None:
        self.pathfinder = pathfinder
        self.factor = factor
        self.reward = None

    def step(self) -> None:
        path_length = self.pathfinder.path_length
        self.reward = path_length * self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
