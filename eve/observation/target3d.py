import numpy as np

from .observation import Observation, gym
from ..target import Target as TargetClass


class Target3D(Observation):
    def __init__(
        self,
        target: TargetClass,
        name: str = "target",
    ) -> None:
        super().__init__(name)
        self.target = target

    @property
    def space(self) -> gym.spaces.Box:
        return self.target.coordinate_space3d

    def step(self) -> None:
        self.obs = np.array(self.target.coordinates3d, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
