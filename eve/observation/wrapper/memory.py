from ..observation import Observation, gym
from . import MemoryResetMode
from typing import Optional
import numpy as np


class Memory(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        n_steps: int,
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
        self.n_steps = n_steps
        self.reset_mode = reset_mode

    @property
    def space(self) -> gym.spaces.Box:
        high = self.wrapped_obs.space.high
        high = np.repeat([high], self.n_steps, axis=0)
        low = self.wrapped_obs.space.low
        low = np.repeat([low], self.n_steps, axis=0)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        new_obs = self.wrapped_obs.obs
        self.obs[1:] = self.obs[:-1]
        self.obs[0] = new_obs

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        new_obs = self.wrapped_obs.obs
        if self.reset_mode == MemoryResetMode.FILL:
            self.obs = np.repeat([new_obs], self.n_steps, axis=0)
        else:
            obs = np.repeat([new_obs], self.n_steps, axis=0) * 0.0
            obs[0] = new_obs
            self.obs = obs
