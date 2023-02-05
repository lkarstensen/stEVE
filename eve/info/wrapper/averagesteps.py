from typing import Dict, Any, Optional
import numpy as np

from .. import Info


class AverageSteps(Info):
    def __init__(
        self,
        wrapped_info: Info,
        n_steps: int,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_info.name
        super().__init__(name)
        self.n_steps = n_steps
        self.wrapped_info = wrapped_info
        self._info = {}
        self._memory: np.ndarray = None

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: np.mean(self._memory, axis=0)}

    def step(self) -> None:
        if self._memory is not None:
            info = self.wrapped_info.info[self.wrapped_info.name]
            np_info = np.array([info])
            self._memory = np.concatenate((np_info, self._memory), axis=0)
        else:
            info = self.wrapped_info.info[self.wrapped_info.name]
            self._memory = np.array([info])

    def reset(self, episode_nr: int = 0) -> None:
        self._memory = None
