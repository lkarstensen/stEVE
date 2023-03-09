from typing import Dict, Any
from .info import Info
from ..target import Target


class TargetReached(Info):
    def __init__(self, target: Target, name: str = "target_reached") -> None:
        super().__init__(name)
        self.target = target

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.target.reached}
