# pylint: disable=unused-argument
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from ..util import EveObject
from .target import Target
from .vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy
from .device import Device


class Intervention(EveObject, ABC):
    vessel_tree: VesselTree
    devices: List[Device]
    fluoroscopy: Fluoroscopy
    target: Target
    stop_device_at_tree_end: bool = True

    @property
    @abstractmethod
    def device_lengths_inserted(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def device_rotations(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def device_lengths_maximum(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def device_diameters(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def reset(
        self,
        episode_number: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def reset_devices(self) -> None:
        ...
