from typing import Dict, Any, List
from . import Info


class Combination(Info):
    def __init__(self, infos: List[Info]) -> None:
        super().__init__("info_combination")
        self.infos = infos

    @property
    def info(self) -> Dict[str, Any]:
        ret_info = {}
        for info in self.infos:
            ret_info.update(info.info)
        return ret_info

    def step(self) -> None:
        for wrapped_info in self.infos:
            wrapped_info.step()

    def reset(self, episode_nr: int = 0) -> None:
        for wrapped_info in self.infos:
            wrapped_info.reset(episode_nr)
