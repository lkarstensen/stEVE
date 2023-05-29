from typing import Union
import numpy as np
from ..tracking2d import Tracking2D
from ..target2d import Target2D
from ...intervention import Intervention
from .normalize import Normalize, Optional


class NormalizeTracking2DEpisode(Normalize):
    def __init__(
        self,
        wrapped_obs: Union[Tracking2D, Target2D],
        intervention: Intervention,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.intervention = intervention
        self._normalization_space = None

    def reset(self, episode_nr: int = 0) -> None:
        self._normalization_space = (
            self.intervention.fluoroscopy.tracking2d_space_episode
        )
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self._normalization_space.low
        high = self._normalization_space.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
