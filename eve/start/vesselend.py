from .start import Start
from ..intervention import Intervention


class VesselEnd(Start):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    def reset(self, episode_nr: int = 0) -> None:
        tip = self.intervention.simulation.instrument_position_vessel_cs[0]
        if self.intervention.vessel_tree.at_tree_end(tip):
            self.intervention.simulation.reset_devices()
