from typing import Dict, Optional
import numpy as np

from ..observation import Observation, gym


class RelativeToLastState(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        self.wrapped_obs = wrapped_obs
        self._last_obs = None

    @property
    def space(self) -> Dict[str, np.ndarray]:
        wrapped_high = self.wrapped_obs.space.high
        wrapped_low = self.wrapped_obs.space.low
        high = wrapped_high - wrapped_low
        low = wrapped_low - wrapped_high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._calc_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._last_obs = self.wrapped_obs()
        self._calc_state()

    def _calc_state(self):
        state = self.wrapped_obs()
        self.obs = state - self._last_obs
        self._last_obs = state
