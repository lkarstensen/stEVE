from abc import ABC, abstractmethod
from typing import Dict, Any

from ..util import EveObject


class Info(EveObject, ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        ...

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr

    def __call__(self) -> Dict[str, Any]:
        return self.info
