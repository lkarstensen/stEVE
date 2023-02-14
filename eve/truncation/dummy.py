from .truncation import Truncation


class Dummy(Truncation):
    @property
    def truncated(self) -> bool:
        return False

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
