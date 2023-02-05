from abc import ABC, abstractmethod

from ..util import EveObject


class Start(EveObject, ABC):
    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        pass
