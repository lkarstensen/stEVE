from abc import ABC, abstractmethod
from ..util import EveObject


class Visualisation(EveObject, ABC):
    @abstractmethod
    def render(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
