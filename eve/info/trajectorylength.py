from typing import Dict, Any

import numpy as np
from .info import Info
from ..intervention import Intervention


class TrajectoryLength(Info):
    def __init__(
        self, intervention: Intervention, name: str = "trajectory length"
    ) -> None:
        super().__init__(name)
        self.intervention = intervention
        self.trajectory_length = 0.0
        self.last_tip_pos = self.intervention.fluoroscopy.tracking3d[0]

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.trajectory_length}

    def step(self) -> None:
        pos = self.intervention.fluoroscopy.tracking3d[0]
        dist = np.linalg.norm(pos - self.last_tip_pos)
        self.trajectory_length += dist
        self.last_tip_pos = pos

    def reset(self, episode_nr: int = 0) -> None:
        self.trajectory_length = 0.0
        self.last_tip_pos = self.intervention.fluoroscopy.tracking3d[0]
