import numpy as np

from .tracking2d import Tracking2D
from .observation import gym


class Tracking3D(Tracking2D):
    @property
    def space(self) -> gym.spaces.Box:
        low = self.intervention.fluoroscopy.tracking3d_space.low
        high = self.intervention.fluoroscopy.tracking3d_space.high
        low = np.tile(low, [self.n_points, 1])
        high = np.tile(high, [self.n_points, 1])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        tracking = self.intervention.fluoroscopy.tracking3d
        self.obs = self._evenly_distributed_tracking(tracking)
