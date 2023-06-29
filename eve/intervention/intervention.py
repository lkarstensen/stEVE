# pylint: disable=unused-argument
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from ..util import EveObject
from .target import Target
from .vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy
from .simulation import Simulation, SimulationMP
from .device import Device


class Intervention(EveObject, ABC):
    vessel_tree: VesselTree
    devices: List[Device]
    fluoroscopy: Fluoroscopy
    target: Target
    stop_device_at_tree_end: bool = True
    normalize_action: bool = False
    simulation: Simulation
    last_action: np.ndarray

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

    def make_mp(self, step_timeout: float = 2, restart_n_resets: int = 200):
        if isinstance(self.simulation, Simulation):
            new_sim = SimulationMP(self.simulation, step_timeout, restart_n_resets)
            self.fluoroscopy.simulation = new_sim
            self.simulation = new_sim

    def make_non_mp(self):
        if isinstance(self.simulation, SimulationMP):
            new_sim = self.simulation.simulation
            self.fluoroscopy.simulation = new_sim
            self.simulation = new_sim

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "devices": self.devices,
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "device_lengths_maximum": self.device_lengths_maximum,
            "device_diameters": self.device_diameters,
            "action_space": self.action_space,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_reset_state(),
            "simulation": self.simulation.get_reset_state(),
            "target": self.target.get_reset_state(),
            "fluoroscopy": self.fluoroscopy.get_reset_state(),
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_step_state(),
            "simulation": self.simulation.get_step_state(),
            "target": self.target.get_step_state(),
            "fluoroscopy": self.fluoroscopy.get_step_state(),
        }
        return deepcopy(state)
