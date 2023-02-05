from .even import Even


class Branching(Even):
    def _calc_interim_targets(self):
        interim_targets = []
        for node in self.pathfinder.path_branching_points:
            interim_targets.append(node.coordinates)
        return interim_targets
