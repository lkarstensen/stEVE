from typing import Optional
import numpy as np
from ..simulation.simulation import Simulation

from .target import Target


class TargetDummy(Target):
    def __init__(self, simulation: Simulation, threshold: float) -> None:
        self.threshold = threshold
        self.simulation = simulation
        self.coordinates_vessel_cs = np.zeros((3,), dtype=np.float32)
        self.reached = False

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.reached = False
