from abc import ABC, abstractmethod
from ..util import EveObject


class Terminal(EveObject, ABC):
    @property
    @abstractmethod
    def terminal(self) -> bool:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...
