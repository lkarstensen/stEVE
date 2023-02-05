import numpy as np

from .centerlinerandom import CenterlineRandom


class BranchEnd(CenterlineRandom):
    def _init_centerline_point_cloud(self):
        self._potential_targets = np.empty((0, 3))
        if self.branch_filter is None:
            branch_keys = self.vessel_tree.keys()
        else:
            branch_keys = set(self.branch_filter) & set(self.vessel_tree.keys())
        for branch in self.vessel_tree:
            if branch.name in branch_keys:
                point = branch.coordinates[-1].reshape(1, -1)
                if self._potential_targets is None:
                    self._potential_targets = point
                else:
                    self._potential_targets = np.concatenate(
                        (self._potential_targets, point), axis=0
                    )
