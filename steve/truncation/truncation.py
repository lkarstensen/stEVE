from abc import ABC, abstractmethod
from ..util import EveObject


class Truncation(EveObject, ABC):
    @property
    @abstractmethod
    def truncated(self) -> bool:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...
