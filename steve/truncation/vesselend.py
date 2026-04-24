from .truncation import Truncation
from ..intervention import Intervention
from ..intervention.vesseltree.vesseltree import at_tree_end
from ..util.coordtransform import tracking3d_to_vessel_cs


class VesselEnd(Truncation):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    @property
    def truncated(self) -> bool:
        tip = self.intervention.fluoroscopy.tracking3d[0]
        tip = tracking3d_to_vessel_cs(
            tip,
            self.intervention.fluoroscopy.image_rot_zx,
            self.intervention.fluoroscopy.image_center,
        )
        return at_tree_end(tip, self.intervention.vessel_tree)

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
