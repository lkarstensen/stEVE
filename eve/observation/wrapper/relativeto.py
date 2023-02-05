from typing import Callable, Optional
import numpy as np

from ..observation import Observation, gym


class RelativeToInsertionPoint(Observation):
    def __init__(
        self,
        relative_to: Callable[[], np.ndarray],
        wrapped_obs: Observation,
        name: Optional[str] = None,
        space: Optional[gym.spaces.Box] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        if wrapped_obs.obs.shape[-1] != 3:
            raise ValueError(
                f"{self.__class__} can only be used with 3 dimensional States.\
                    Not with {wrapped_obs.obs.shape[-1]} Dimensions"
            )
        self.realtive_to = relative_to
        self.wrapped_obs = wrapped_obs
        self._space = space or self._calc_space()

        self._realtive_to_called: np.ndarray = None

    @property
    def space(self) -> gym.spaces.Box:
        return self._space

    def step(self) -> None:
        self.wrapped_obs.step()
        self._realtive_to_called = self.realtive_to()
        self._calc_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._realtive_to_called = self.realtive_to()
        self._calc_state()

    def _calc_state(self):
        state = self.wrapped_obs()
        subtrahend = np.full_like(state, self._realtive_to_called, dtype=np.float32)
        self.obs = state - subtrahend

    def _calc_space(self) -> gym.spaces.Box:
        wrapped_high = self.wrapped_obs.space.high
        called = self.realtive_to()
        called = np.full_like(wrapped_high, called, dtype=np.float32)
        high = wrapped_high - called
        wrapped_low = self.wrapped_obs.space.low
        low = wrapped_low - called
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
