from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..util import EveObject
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_2d


class InterimTarget(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset().
    # Coordinates are in the tracking coordinate space
    all_coordinates3d: List[np.ndarray]
    reached: bool
    threshold: float
    intervention: Intervention

    @property
    def coordinates3d(self) -> np.ndarray:
        return self.all_coordinates3d[0] if len(self.all_coordinates3d) > 0 else None

    @property
    def coordinates2d(self) -> np.ndarray:
        if self.coordinates3d is None:
            return None
        return tracking3d_to_2d(self.coordinates3d)

    @property
    def all_coordinates2d(self) -> List[np.ndarray]:
        if not self.all_coordinates3d:
            return []
        return tracking3d_to_2d(self.all_coordinates3d)

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...
