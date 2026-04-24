import numpy as np

from .observation import Observation, gym
from ..intervention import Intervention


class Target3D(Observation):
    def __init__(
        self,
        intervention: Intervention,
        name: str = "target3d",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.fluoroscopy.tracking3d_space

    def step(self) -> None:
        self.obs = np.array(self.intervention.target.coordinates3d, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
