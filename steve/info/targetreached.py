from typing import Dict, Any
from .info import Info
from ..intervention import Intervention


class TargetReached(Info):
    def __init__(
        self, intervention: Intervention, name: str = "target_reached"
    ) -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.intervention.target.reached}
