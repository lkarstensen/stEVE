from .interimtarget import InterimTarget


class Dummy(InterimTarget):
    def __init__(self) -> None:
        self.all_coordinates3d = []
        self.reached = False
        self.threshold = 1

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
