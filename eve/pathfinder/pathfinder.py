from abc import ABC, abstractmethod
import numpy as np
from ..util import EveObject
from ..intervention import Intervention


class Pathfinder(EveObject, ABC):
    path_length: float
    path_points3d: np.ndarray
    path_branching_points3d: np.ndarray
    intervention: Intervention

    @property
    def path_points2d(self) -> np.ndarray:
        return self.intervention.fluoroscopy.tracking3d_to_2d(self.path_points3d)

    @property
    def path_branching_points2d(self) -> np.ndarray:
        return self.intervention.fluoroscopy.tracking3d_to_2d(
            self.path_branching_points3d
        )

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self, episode_nr=0) -> None:
        pass
