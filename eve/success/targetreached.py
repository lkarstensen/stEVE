from .success import Success
from ..target import Target


class TargetReached(Success):
    def __init__(self, target: Target) -> None:
        self.target = target

    @property
    def success(self) -> float:
        return float(self.target.reached)
