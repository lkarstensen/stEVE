from .start import Start
from ..intervention import Intervention


class VesselEnd(Start):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    def reset(self, episode_nr: int = 0) -> None:
        tip = self.intervention.fluoroscopy.tracking3d[0]
        tip_vessel_cs = self.intervention.fluoroscopy.tracking3d_to_vessel_cs(tip)
        if self.intervention.vessel_tree.at_tree_end(tip_vessel_cs):
            self.intervention.simulation.reset_devices()
