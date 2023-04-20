from typing import Optional, List
import random
import numpy as np

from .target import Target
from ..vesseltree import VesselTree
from ..intervention.intervention import Intervention


class CenterlineRandom(Target):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        threshold: float,
        branches: Optional[List[str]] = None,
        min_distance_between_possible_targets: Optional[float] = None,
    ) -> None:
        self.intervention = intervention
        self.threshold = threshold
        self.vessel_tree = vessel_tree
        self.branches = branches
        self.min_distance_between_possible_targets = (
            min_distance_between_possible_targets
        )

        self._potential_targets = None
        self._branches_initialized = None
        self._rng = random.Random()

    def reset(self, episode_nr=0, seed=None) -> None:
        super().reset(episode_nr, seed)
        if seed is not None:
            self._rng = random.Random(seed)
        if self._branches_initialized != self.vessel_tree.branches:
            self._init_centerline_point_cloud()
            self._branches_initialized = self.vessel_tree.branches
        self.coordinates_vessel_cs = self._rng.choice(self._potential_targets)
        self.reached = False

    def _init_centerline_point_cloud(self):
        if self.branches is None:
            branch_keys = self.vessel_tree.keys()
            excluded_branches = []
        else:
            branch_keys = set(self.branches) & set(self.vessel_tree.keys())
            excluded_branches = set(self.vessel_tree.keys()) - set(self.branches)
        branch_keys = sorted(branch_keys)
        potential_targets = np.empty((0, 3))
        for branch in branch_keys:
            points = self.vessel_tree[branch].coordinates
            potential_targets = np.vstack((potential_targets, points))

        in_excluded = self._in_excluded_branches(potential_targets, excluded_branches)
        outside_forbidden = np.invert(in_excluded)
        self._potential_targets = potential_targets[outside_forbidden]

    def _in_excluded_branches(
        self, coordinates: np.ndarray, excluded_branches: List[str]
    ):
        in_branch = [False] * coordinates.shape[0]
        for branch_name in excluded_branches:
            branch = self.vessel_tree[branch_name]
            in_branch = branch.in_branch(coordinates) + in_branch
        return in_branch
