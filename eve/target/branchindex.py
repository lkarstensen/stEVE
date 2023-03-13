import numpy as np

from .target import Target, gym
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
        super().__init__(intervention, threshold)
        self.vessel_tree = vessel_tree
        self.branch = branch
        self.idx = idx

        self.coordinates = None
        self.reached = False
        self._branches = None

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self._branches != self.vessel_tree.branches:
            self.coordinates = self.vessel_tree[self.branch].coordinates[self.idx]
            self._branches = self.vessel_tree.branches
        self.reached = False
