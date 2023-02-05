from typing import Optional
import numpy as np
from ..observation import Observation, gym


class Normalize(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        self.wrapped_obs = wrapped_obs

    @property
    def space(self) -> gym.spaces.Box:
        wrapped_high = self.wrapped_obs.space.high
        high = self._normalize(wrapped_high)
        wrapped_low = self.wrapped_obs.space.low
        low = self._normalize(wrapped_low)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        new_obs = self.wrapped_obs()
        self.obs = self._normalize(new_obs)

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        new_obs = self.wrapped_obs()
        self.obs = self._normalize(new_obs)

    def _normalize(self, obs) -> np.ndarray:
        low = self.wrapped_obs.space.low
        high = self.wrapped_obs.space.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
