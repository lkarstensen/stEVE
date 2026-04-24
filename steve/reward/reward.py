from abc import ABC, abstractmethod

from ..util import EveObject


class Reward(EveObject, ABC):
    reward = 0.0

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr: int = 0) -> None:
        ...

    def __call__(self) -> int:
        return self.reward
