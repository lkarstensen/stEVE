from typing import Optional
import numpy as np

from .observation import Observation, gym
from ..intervention import Intervention
from ..interimtarget import InterimTarget


class Target2D(Observation):
    def __init__(
        self,
        intervention: Intervention,
        interim_target: Optional[InterimTarget] = None,
        name: str = "target2d",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.interim_target = interim_target
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.fluoroscopy.tracking2d_space

    def step(self) -> None:
        if (
            self.interim_target is not None
            and self.interim_target.coordinates2d is not None
        ):
            self.obs = np.array(self.interim_target.coordinates2d, dtype=np.float32)
        else:
            self.obs = np.array(
                self.intervention.target.coordinates2d, dtype=np.float32
            )

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
