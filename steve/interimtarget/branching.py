from .even import Even


class Branching(Even):
    def _calc_interim_targets(self):
        return self.pathfinder.path_branching_points3d.copy()
