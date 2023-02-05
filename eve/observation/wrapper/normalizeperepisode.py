from typing import Callable
import numpy as np
from .normalize import Normalize, Observation, Optional


class NormalizePerEpisode(Normalize):
    def __init__(
        self,
        wrapped_obs: Observation,
        low_callable: Callable[[], np.ndarray],
        high_callable: Callable[[], np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.low_callable = low_callable
        self.high_callable = high_callable
        self._low = low_callable()
        self._high = high_callable()

    def reset(self, episode_nr: int = 0) -> None:
        self._low = self.low_callable()
        self._high = self.high_callable()
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self._low
        high = self._high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
