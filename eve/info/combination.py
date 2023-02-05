from typing import Dict, Any, List
from . import Info


class Combination(Info):
    def __init__(self, infos: List[Info]) -> None:
        super().__init__("info_combination")
        self.infos = infos
        self._info = {}

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    def step(self) -> None:
        self._info = {}
        for wrapped_info in self.infos:
            wrapped_info.step()
            self._info.update(wrapped_info())

    def reset(self, episode_nr: int = 0) -> None:
        self._info = {}
        for wrapped_info in self.infos:
            wrapped_info.reset(episode_nr)
            self._info.update(wrapped_info())
