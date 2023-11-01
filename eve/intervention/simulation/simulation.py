from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ...util import EveObject
from ..device import Device


class Simulation(EveObject, ABC):
    simulation_error: bool

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
    def step(self, action: np.ndarray, duration: float):
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

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "dof_positions": self.dof_positions,
            "inserted_lengths": self.inserted_lengths,
            "rotations": self.rotations,
            "simulation_error": self.simulation_error,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        return self.get_reset_state()


class SimulationDummy(Simulation):
    def __init__(self) -> None:
        self.simulation_error = False
        self._n_devices = 1

    @property
    def dof_positions(self) -> np.ndarray:
        return np.zeros((50, 3))

    @property
    def inserted_lengths(self) -> List[float]:
        return [0.0] * self._n_devices

    @property
    def rotations(self) -> List[float]:
        return [0.0] * self._n_devices

    def close(self):
        ...

    def step(self, action: np.ndarray, duration: float):
        ...

    def reset_devices(self):
        ...

    def add_interim_targets(self, positions: List[Tuple[float, float, float]]):
        ...

    def remove_interim_target(self, interim_target):
        ...

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
        self._n_devices = len(devices)
