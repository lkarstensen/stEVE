from copy import deepcopy
from typing import Optional
import numpy as np

from ..observation import Observation, gym


class RelativeToFirstRow(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        name: Optional[str] = None,
    ) -> None:
        self.name = name or wrapped_obs.name
        self.wrapped_obs = wrapped_obs
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        wrapped_high = self.wrapped_obs.space.high
        wrapped_low = self.wrapped_obs.space.low
        high = deepcopy(wrapped_high)
        subtrahend = np.broadcast_to(wrapped_low[0], high[1:].shape)
        high[1:] = high[1:] - subtrahend
        low = deepcopy(wrapped_low)
        subtrahend = np.broadcast_to(wrapped_high[0], low[1:].shape)
        low[1:] = low[1:] - subtrahend
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._calc_obs()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._calc_obs()

    def _calc_obs(self):
        wrapped_obs = self.wrapped_obs()
        subtrahend = np.full(wrapped_obs.shape, wrapped_obs[0])
        subtrahend[0] *= 0.0
        self.obs = wrapped_obs - subtrahend
