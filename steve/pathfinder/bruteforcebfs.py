from typing import Dict, Generator, List, Tuple, NamedTuple
from copy import deepcopy
from math import inf
import numpy as np

from .pathfinder import Pathfinder
from ..intervention.vesseltree import (
    Branch,
    BranchingPoint,
    find_nearest_branch_to_point,
)

from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs, vessel_cs_to_tracking3d


def get_length(path: np.ndarray):
    return np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1))


class BPConnection(NamedTuple):
    length: float
    points: np.ndarray


class BruteForceBFS(Pathfinder):
    def __init__(self, intervention: Intervention):
        self.intervention = intervention
        self.path_length: float = 0.0
        self.path_points3d: np.ndarray = np.empty((0, 3))
        self.path_branching_points3d: np.ndarray = np.empty((0, 3))
        self._branches = None
        self._node_connections = None
        self._search_graph_base = None

    def reset(self, episode_nr=0) -> None:
        if self._branches != self.intervention.vessel_tree.branches:
            self._init_vessel_tree()
            self.path_length = 0.0
            self.path_points3d = np.empty((0, 3))
            self.path_branching_points3d = np.empty((0, 3))
            self._branches = self.intervention.vessel_tree.branches
        self.step()

    def step(self) -> None:
        fluoro = self.intervention.fluoroscopy
        position = fluoro.tracking3d[0]
        position_vessel_cs = tracking3d_to_vessel_cs(
            position, fluoro.image_rot_zx, fluoro.image_center
        )
        target = self.intervention.target.coordinates3d
        target_vessel_cs = tracking3d_to_vessel_cs(
            target, fluoro.image_rot_zx, fluoro.image_center
        )
        position_branch = find_nearest_branch_to_point(
            position_vessel_cs, self.intervention.vessel_tree
        )
        target_branch = find_nearest_branch_to_point(
            target_vessel_cs, self.intervention.vessel_tree
        )

        (
            path_branching_points,
            self.path_length,
            path_points,
        ) = self._get_shortest_path(
            position_branch, target_branch, position_vessel_cs, target_vessel_cs
        )
        if path_branching_points is not None:
            path_branching_points = [
                branching_point.coordinates for branching_point in path_branching_points
            ]
            path_branching_points = np.array(path_branching_points)
            self.path_branching_points3d = vessel_cs_to_tracking3d(
                path_branching_points,
                fluoro.image_rot_zx,
                fluoro.image_center,
                fluoro.field_of_view,
            )
        else:
            self.path_branching_points3d = None
        self.path_points3d = vessel_cs_to_tracking3d(
            path_points,
            fluoro.image_rot_zx,
            fluoro.image_center,
            fluoro.field_of_view,
        )

    def _init_vessel_tree(self) -> None:
        self._node_connections = self._initialize_node_connections(
            self.intervention.vessel_tree.branching_points
        )
        self._search_graph_base = self._initialize_search_graph_base()

    def _initialize_node_connections(
        self, branching_points: Tuple[BranchingPoint]
    ) -> Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]]:
        node_connections = {}
        for branching_point in branching_points:
            node_connections[branching_point] = {}
            for connection in branching_point.connections:
                for target_branching_point in branching_points:
                    if branching_point == target_branching_point:
                        continue
                    if connection in target_branching_point.connections:
                        points = connection.get_path_along_branch(
                            branching_point.coordinates,
                            target_branching_point.coordinates,
                        )
                        length = get_length(points)

                        node_connections[branching_point][
                            target_branching_point
                        ] = BPConnection(length, points)
        return node_connections

    def _initialize_search_graph_base(
        self,
    ) -> Dict[BranchingPoint, List[BranchingPoint]]:
        _search_graph_base = {}
        for node in self._node_connections:
            _search_graph_base[node] = list(self._node_connections[node].keys())
        return _search_graph_base

    def _get_shortest_path(
        self,
        start_branch: Branch,
        target_branch: Branch,
        start: np.ndarray,
        target: np.ndarray,
    ):  # -> Tuple[List[BranchingPoint], float, List[CenterlinePoint]]:
        search_graph = self._create_search_graph(start_branch, target_branch)
        bfs_paths = self._get_bfs_paths_generator(search_graph)

        shortest_path_length = inf
        shortest_path = None

        path = next(bfs_paths, None)
        if path is None:
            shortest_path_points = np.empty((1, 3))
            shortest_path_length = 0.0

        elif len(path) == 2:
            shortest_path_points = start_branch.get_path_along_branch(start, target)
            shortest_path_length = get_length(shortest_path_points)

        else:
            shortest_path_points = start_branch.get_path_along_branch(
                start, path[1].coordinates
            )
            shortest_path_length = get_length(shortest_path_points)

            for node, next_node in zip(path[1:-2], path[2:-1]):
                connection = self._node_connections[node][next_node]
                shortest_path_length += connection.length
                shortest_path_points = np.vstack(
                    (shortest_path_points, connection.points[1:])
                )

            target_points = target_branch.get_path_along_branch(
                path[-2].coordinates, target
            )

            target_length = get_length(target_points)
            shortest_path_points = np.vstack((shortest_path_points, target_points[1:]))
            shortest_path_length += target_length
            shortest_path = path[1:-1]

        return shortest_path, shortest_path_length, shortest_path_points

    def _create_search_graph(self, start_branch, target_branch):
        search_graph = deepcopy(self._search_graph_base)
        if start_branch == target_branch:
            search_graph["start"] = ["target"]
            return search_graph

        start_connections = []
        for branching_point in self.intervention.vessel_tree.branching_points:
            if start_branch in branching_point.connections:
                start_connections.append(branching_point)
            if target_branch in branching_point.connections:
                search_graph[branching_point].append("target")

        search_graph["start"] = start_connections

        return search_graph

    def _get_bfs_paths_generator(
        self, graph: Dict
    ) -> Generator[List[BranchingPoint], None, None]:
        """bfs path search

        Arguments:
            graph {dict} -- dict of nodes containing all connected nodes
                including the entries 'start' and 'target'

        Yields:
            [list] -- a list of the nodes along the path with the node
                names as entries
        """
        queue = [("start", ["start"])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next_bp in graph[vertex]:
                if next_bp in path:
                    continue
                elif next_bp == "target":
                    yield path + [next_bp]
                else:
                    queue.append((next_bp, path + [next_bp]))
