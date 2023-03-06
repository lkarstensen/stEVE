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

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        return self.wrapped_target.coordinate_space

    def reset(self, episode_nr: int = 0) -> None:
        while True:
            self.wrapped_target.reset(episode_nr)
            if not self._in_forbidden_branches(self.wrapped_target.coordinates):
                break
        self.coordinates = self.wrapped_target.coordinates

    def _in_forbidden_branches(self, coordinates: np.ndarray):
        in_branch = False
        for branch_name in self.branches_to_avoid:
            branch = self.vessel_tree[branch_name]
            in_branch = branch.in_branch(coordinates) or in_branch
        return in_branch
