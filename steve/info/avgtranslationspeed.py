from typing import Dict, Any
from .info import Info
from ..intervention import Intervention


class AverageTranslationSpeed(Info):
    def __init__(
        self, intervention: Intervention, name: str = "average translation speed"
    ) -> None:
        super().__init__(name)
        self.intervention = intervention
        self.trans_speeds = []

    @property
    def info(self) -> Dict[str, Any]:
        value = (
            sum(self.trans_speeds) / len(self.trans_speeds)
            if self.trans_speeds
            else 0.0
        )
        return {self.name: value}

    def step(self) -> None:
        self.trans_speeds.append(abs(self.intervention.last_action[0, 0]))

    def reset(self, episode_nr: int = 0) -> None:
        self.trans_speeds = []
