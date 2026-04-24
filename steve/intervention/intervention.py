# pylint: disable=unused-argument
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from ..util import EveObject
from .target import Target
from .vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy, SimulatedFluoroscopy
from .simulation import Simulation, SimulationMP
from .instrument import Instrument


class Intervention(EveObject, ABC):
    instruments: List[Instrument]
    vessel_tree: VesselTree
    fluoroscopy: Fluoroscopy
    target: Target
    normalize_action: bool = False
    last_action: np.ndarray
    instrument_lengths_inserted: List[float]
    instrument_rotations: List[float]
    instrument_lengths_maximum: List[float]
    instrument_diameters: List[float]
    action_space: gym.spaces.Box

    @abstractmethod
    def step(self, action: np.ndarray) -> None: ...

    @abstractmethod
    def reset(
        self,
        episode_number: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def reset_instruments(self) -> None: ...

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "instruments": self.instruments,
            "instrument_lengths_inserted": self.instrument_lengths_inserted,
            "instrument_rotations": self.instrument_rotations,
            "instrument_lengths_maximum": self.instrument_lengths_maximum,
            "instrument_diameters": self.instrument_diameters,
            "action_space": self.action_space,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_reset_state(),
            "target": self.target.get_reset_state(),
            "fluoroscopy": self.fluoroscopy.get_reset_state(),
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "instrument_lengths_inserted": self.instrument_lengths_inserted,
            "instrument_rotations": self.instrument_rotations,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_step_state(),
            "target": self.target.get_step_state(),
            "fluoroscopy": self.fluoroscopy.get_step_state(),
        }
        return deepcopy(state)


class SimulatedIntervention(Intervention, ABC):
    fluoroscopy: SimulatedFluoroscopy
    simulation: Simulation
    stop_instrument_at_tree_end: bool = True

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
            "instruments": self.instruments,
            "instrument_lengths_inserted": self.instrument_lengths_inserted,
            "instrument_rotations": self.instrument_rotations,
            "instrument_lengths_maximum": self.instrument_lengths_maximum,
            "instrument_diameters": self.instrument_diameters,
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
            "instrument_lengths_inserted": self.instrument_lengths_inserted,
            "instrument_rotations": self.instrument_rotations,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_step_state(),
            "simulation": self.simulation.get_step_state(),
            "target": self.target.get_step_state(),
            "fluoroscopy": self.fluoroscopy.get_step_state(),
        }
        return deepcopy(state)
