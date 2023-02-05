from ..observation import Observation, gym

from typing import Optional

import numpy as np


class CoordinatesTo2D(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        dim_to_delete: str,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        self.dim_to_delete = dim_to_delete
        if dim_to_delete == "y":
            self._delete_idx = 1
        elif dim_to_delete == "z":
            self._delete_idx = 2
        elif dim_to_delete == "x":
            self._delete_idx = 0
        else:
            raise ValueError(f"{dim_to_delete = } is invalid. Needs to be x,y or z")
        self.wrapped_obs = wrapped_obs

    @property
    def space(self) -> gym.spaces.Box:
        high = self.wrapped_obs.space.high
        high = np.delete(high, self._delete_idx, axis=-1)
        low = self.wrapped_obs.space.low
        low = np.delete(low, self._delete_idx, axis=-1)
        return gym.spaces.Box(low=low, high=high)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._get_obs()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._get_obs()

    def _get_obs(self):
        obs = self.wrapped_obs.obs
        obs = np.delete(obs, self._delete_idx, axis=-1)
        self.obs = obs
