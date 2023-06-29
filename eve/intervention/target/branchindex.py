from .target import Target
from ..vesseltree import VesselTree
from ..fluoroscopy import Fluoroscopy
from ...util.coordtransform import vessel_cs_to_tracking3d


class BranchIndex(Target):
    def __init__(
        self,
        vessel_tree: VesselTree,
        fluoroscopy: Fluoroscopy,
        threshold: float,
        branch: str,
        idx: int,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.threshold = threshold
        self.branch = branch
        self.idx = idx

        self.coordinates3d = None
        self.reached = False

    def reset(self, episode_nr=0, seed=None) -> None:
        target_vessel_cs = self.vessel_tree[self.branch].coordinates[self.idx]
        self.coordinates3d = vessel_cs_to_tracking3d(
            target_vessel_cs,
            self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center,
            self.fluoroscopy.field_of_view,
        )
        self.reached = False
