from .target import Target
from ..vesseltree import VesselTree
from ..fluoroscopy import Fluoroscopy


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
        self.coordinates3d = self.fluoroscopy.vessel_cs_to_tracking3d(target_vessel_cs)
        self.reached = False
