from typing import Dict, Any
from .info import Info
from ..pathfinder import Pathfinder


class PathRatio(Info):
    def __init__(self, pathfinder: Pathfinder, name: str = "path_ratio") -> None:
        super().__init__(name)
        self.pathfinder = pathfinder
        self.initial_pathlength = 1

    @property
    def info(self) -> Dict[str, Any]:
        portion = 1 - (self.pathfinder.path_length / self.initial_pathlength)
        return {self.name: portion}

    def reset(self, episode_nr: int = 0) -> None:
        self.initial_pathlength = self.pathfinder.path_length
