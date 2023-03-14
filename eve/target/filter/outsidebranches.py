import random
from typing import List
from ..target import Target, gym, np
from ...vesseltree import VesselTree


class OutsideBranches(Target):
    def __init__(
        self,
        wrapped_target: Target,
        vessel_tree: VesselTree,
        branches_to_avoid: List[str],
    ) -> None:
        super().__init__(wrapped_target.intervention, wrapped_target.threshold)

        self.wrapped_target = wrapped_target
        self.vessel_tree = vessel_tree
        self.branches_to_avoid = branches_to_avoid
        self.coordinates = np.zeros((3,), dtype=np.float32)
        self._rng = random.Random()
        self._initialized_potential_targets = None
        self.potential_targets = None

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        return self.wrapped_target.coordinate_space

    def reset(self, episode_nr: int = 0, seed=None) -> None:
        super().reset(episode_nr, seed)
        self.wrapped_target.reset(episode_nr, seed)
        if seed is not None:
            self._rng = random.Random(seed)
        if np.any(
            self._initialized_potential_targets != self.wrapped_target.potential_targets
        ):
            pot_targets = self.wrapped_target.potential_targets
            in_forbidden = self._in_forbidden_branches(pot_targets)
            outside_forbidden = np.invert(in_forbidden)
            self.potential_targets = pot_targets[outside_forbidden]
            self._initialized_potential_targets = self.wrapped_target.potential_targets

        self.coordinates = self._rng.choice(self.potential_targets)
        self.reached = False

    def _in_forbidden_branches(self, coordinates: np.ndarray):
        in_branch = [False] * coordinates.shape[0]
        for branch_name in self.branches_to_avoid:
            branch = self.vessel_tree[branch_name]
            in_branch = branch.in_branch(coordinates) + in_branch
        return in_branch
