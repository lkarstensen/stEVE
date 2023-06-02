from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from ...util import EveObject
from ..device import Device


class Simulation(EveObject, ABC):
    @property
    @abstractmethod
    def dof_positions(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def inserted_lengths(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def rotations(self) -> List[float]:
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def do_steps(self, action: np.ndarray, duration: float):
        ...

    @abstractmethod
    def reset_devices(self):
        ...

    @abstractmethod
    def add_interim_targets(self, positions: List[Tuple[float, float, float]]):
        ...

    @abstractmethod
    def remove_interim_target(self, interim_target):
        ...

    @abstractmethod
    def reset(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        devices: List[Device],
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
        vessel_visual_path: Optional[str] = None,
        seed: int = None,
    ):
        ...
