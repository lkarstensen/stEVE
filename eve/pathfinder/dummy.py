import numpy as np

from .pathfinder import Pathfinder


class Dummy(Pathfinder):
    def __init__(self, *args, **kwds) -> None:
        pass

    @property
    def path_length(self) -> float:
        return 0.0

    @property
    def path_points(self) -> np.ndarray:
        return []

    @property
    def path_branching_points(self):  # -> List[BranchingPoint]:
        return []

    def step(self) -> None:
        pass

    def reset(self, episode_nr=0) -> None:
        pass
