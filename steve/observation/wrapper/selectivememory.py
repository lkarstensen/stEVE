from typing import List, Optional
import numpy as np

from ..observation import Observation, gym
from . import MemoryResetMode


class SelectiveMemory(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        steps: List[int],
        reset_mode: MemoryResetMode,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        assert reset_mode in [
            0,
            1,
        ], f"Reset mode must be 'MemoryResetMode.FILL' or 'MemoryResetMode.ZERO'. {reset_mode} is not possible"
        self.wrapped_obs = wrapped_obs
        self.steps = steps
        self.reset_mode = reset_mode
        self._memory = None

    @property
    def space(self) -> gym.spaces.Box:
        high = self.wrapped_obs.space.high
        high = np.repeat([high], len(self.steps), axis=0)
        low = self.wrapped_obs.space.low
        low = np.repeat([low], len(self.steps), axis=0)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        new_obs = self.wrapped_obs()
        self._memory[1:] = self._memory[:-1]
        self._memory[0] = new_obs
        self._calc_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        new_obs = self.wrapped_obs()
        if self.reset_mode == MemoryResetMode.FILL:
            self._memory = np.repeat([new_obs], max(self.steps), axis=0)
        else:
            memory = np.repeat([new_obs], max(self.steps), axis=0) * 0.0
            memory[0] = new_obs
            self._memory = memory
        self._calc_state()

    def _calc_state(self):
        state = [self._memory[timestep] for timestep in self.steps]
        self.obs = np.array(state)
