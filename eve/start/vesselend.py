from .start import Start
from ..intervention import Intervention
from ..intervention.vesseltree.vesseltree import at_tree_end
from ..util.coordtransform import tracking3d_to_vessel_cs


class VesselEnd(Start):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    def reset(self, episode_nr: int = 0) -> None:
        tip = self.intervention.fluoroscopy.tracking3d[0]
        tip_vessel_cs = tracking3d_to_vessel_cs(
            tip,
            self.intervention.fluoroscopy.image_rot_zx,
            self.intervention.fluoroscopy.image_center,
        )
        if at_tree_end(tip_vessel_cs, self.intervention.vessel_tree):
            self.intervention.simulation.reset_devices()
