import numpy as np
from ..tracking import Tracking
from .normalize import Normalize, Optional


class NormalizeTrackingPerEpisode(Normalize):
    def __init__(
        self,
        wrapped_obs: Tracking,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self._normalization_space = wrapped_obs.intervention.tracking_space_episode

    def reset(self, episode_nr: int = 0) -> None:
        self._normalization_space = self.wrapped_obs.intervention.tracking_space_episode
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self._normalization_space.low
        high = self._normalization_space.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
