from .target import Target
from ..vesseltree import VesselTree
from ..intervention.intervention import Intervention


class BranchIndex(Target):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        threshold: float,
        branch: str,
        idx: int,
    ) -> None:
        self.intervention = intervention
        self.threshold = threshold
        self.vessel_tree = vessel_tree
        self.branch = branch
        self.idx = idx

        self.coordinates2d = None
        self.reached = False

    def reset(self, episode_nr=0, seed=None) -> None:
        self.coordinates_vessel_cs = self.vessel_tree[self.branch].coordinates[self.idx]
        self.reached = False
