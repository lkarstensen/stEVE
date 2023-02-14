from typing import Optional
import numpy as np

from ..observation import Observation, gym
from ...intervention.intervention import Intervention


class RelativeToTip(Observation):
    def __init__(
        self,
        intervention: Intervention,
        wrapped_obs: Observation,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        if wrapped_obs.space.shape[-1] != 3:
            raise ValueError(
                f"{self.__class__} can only be used with 3 dimensional States. Not with {wrapped_obs.space.shape[-1]} Dimensions"
            )
        self.intervention = intervention
        self.wrapped_obs = wrapped_obs
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        wrapped_high = self.wrapped_obs.space.high
        tip_low = self.intervention.tracking_space.low
        tip_low = np.full_like(wrapped_high, tip_low, dtype=np.float32)
        high = wrapped_high - tip_low
        wrapped_low = self.wrapped_obs.low[self.wrapped_obs.name]
        tip_high = self.intervention.tracking_space.high
        tip_high = np.full_like(wrapped_low, tip_high, dtype=np.float32)
        low = wrapped_low - tip_high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._calc_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._calc_state()

    def _calc_state(self):
        state = self.wrapped_obs()
        tip = self.intervention.tracking[0]
        subtrahend = np.full_like(state, tip)
        self.obs = state - subtrahend
