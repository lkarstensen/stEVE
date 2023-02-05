import numpy as np

from .pathfinder import Pathfinder, gym


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

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        low = np.zeros((3,), dtype=np.float32)
        high = low
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        pass

    def reset(self, episode_nr=0) -> None:
        pass
