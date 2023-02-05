from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym

from .vesseltree import VesselTree


@dataclass(frozen=True)
class Device(ABC):
    name: str
    length: float
    velocity_limit: Tuple[float, float]


class Intervention(ABC):
    def __init__(
        self,
        vessel_tree: VesselTree,
        image_frequency: float,
        dt_simulation: float,
        velocity_limits: List[Tuple[float, float]],
    ) -> None:
        self.vessel_tree = vessel_tree
        self.image_frequency = image_frequency
        self.dt_simulation = dt_simulation
        self.velocity_limits = np.array(velocity_limits, dtype=np.float32)
        self.last_action = np.zeros_like(self.velocity_limits, dtype=np.float32)
        self.simulation_error = False

    @property
    def tracking(self) -> np.ndarray:
        return self.tracking_ground_truth

    @property
    @abstractmethod
    def tracking_ground_truth(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def tracking_per_device(self) -> Dict[Device, np.ndarray]:
        ...

    @property
    @abstractmethod
    def device_lengths_inserted(self) -> Dict[Device, float]:
        ...

    @property
    @abstractmethod
    def device_lengths_maximum(self) -> Dict[Device, float]:
        ...

    @property
    @abstractmethod
    def device_rotations(self) -> Dict[Device, float]:
        ...

    @property
    @abstractmethod
    def device_diameters(self) -> Dict[Device, float]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: int = None) -> None:
        ...

    @abstractmethod
    def reset_devices(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    def tracking_space(self) -> gym.spaces.Box:
        return self.vessel_tree.coordinate_space

    @property
    def action_space(self) -> gym.spaces.Box:
        low = -self.velocity_limits
        high = self.velocity_limits
        return gym.spaces.Box(low=low, high=high)
