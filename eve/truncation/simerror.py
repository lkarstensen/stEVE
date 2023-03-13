from .truncation import Truncation
from ..intervention import Intervention


class SimError(Truncation):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    @property
    def truncated(self) -> bool:
        return self.intervention.simulation_error

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
