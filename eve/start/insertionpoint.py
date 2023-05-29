from .start import Start
from ..intervention import Intervention


class InsertionPoint(Start):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    def reset(self, episode_nr: int = 0) -> None:
        self.intervention.simulation.reset_devices()
