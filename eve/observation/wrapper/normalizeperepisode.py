from typing import Callable
import numpy as np
import gymnasium as gym
from .normalize import Normalize, Observation, Optional


class NormalizePerEpisode(Normalize):
    def __init__(
        self,
        wrapped_obs: Observation,
        normalize_space_callable: Callable[[], gym.spaces.Box],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.normalize_space_callable = normalize_space_callable
        self._normalize_space = normalize_space_callable()

    def reset(self, episode_nr: int = 0) -> None:
        self._normalize_space = self.normalize_space_callable()
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self._normalize_space.low
        high = self._normalize_space.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
