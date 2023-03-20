from .start import Start
from ..intervention.intervention import Intervention
from ..vesseltree import VesselTree


class VesselEnd(Start):
    def __init__(self, intervention: Intervention, vessel_tree: VesselTree) -> None:
        self.intervention = intervention
        self.vessel_tree = vessel_tree

    def reset(self, episode_nr: int = 0) -> None:
        tip = self.intervention.instrument_position_vessel_cs[0]
        if self.vessel_tree.at_tree_end(tip):
            self.intervention.reset_devices()
