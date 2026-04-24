from typing import Dict, Any, Optional
import numpy as np

from .. import Info


class AverageEpisodesLastStep(Info):
    def __init__(
        self,
        wrapped_info: Info,
        n_episodes: int,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_info.name
        super().__init__(name)
        self.n_episodes = n_episodes
        self.wrapped_info = wrapped_info
        self._info = {}
        self._memory: np.ndarray = None
        self._last_step_info = None

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: np.mean(self._memory, axis=0)}

    def step(self) -> None:
        self._last_step_info = self.wrapped_info.info[self.wrapped_info.name]

    def reset(self, episode_nr: int = 0) -> None:
        if episode_nr == 0:
            ...
        elif episode_nr == 1:
            self._memory = np.array([self._last_step_info])

        else:
            np_info = np.array([self._last_step_info])
            self._memory = np.concatenate((np_info, self._memory), axis=0)
