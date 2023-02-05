from abc import ABC, abstractmethod

from ..util import EveObject


class Success(EveObject, ABC):
    def step(self) -> None:
        ...

    def reset(self, *args, **kwds) -> None:
        _ = args
        _ = kwds

    @property
    @abstractmethod
    def success(self) -> float:
        ...
