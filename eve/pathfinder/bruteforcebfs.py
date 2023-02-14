from typing import Dict, Generator, List, Tuple, NamedTuple
from copy import deepcopy
from math import inf
import numpy as np

from .pathfinder import Pathfinder, gym
from ..vesseltree import Branch, VesselTree, BranchingPoint
from ..intervention.intervention import Intervention
from ..target import Target


def get_length(path: np.ndarray):
    return np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1))


class BPConnection(NamedTuple):
    length: float
    points: np.ndarray


class BruteForceBFS(Pathfinder):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        target: Target,
    ):
        self.vessel_tree = vessel_tree
        self.intervention = intervention
        self.target = target
        self._path_length: float = 0.0
        self._path_points: np.ndarray = np.empty((0, 3))
        self._path_branching_points: np.ndarray = np.empty((0, 3))
        self._branches = None
        self._node_connections = None
        self._search_graph_base = None

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def path_points(self) -> np.ndarray:
        return self._path_points

    @property
    def path_branching_points(self):  # -> List[BranchingPoint]:
        return self._path_branching_points

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr=0) -> None:
        if self._branches != self.vessel_tree.branches:
            self._init_vessel_tree()
            self._path_length = 0.0
            self._path_points = np.empty((0, 3))
            self._path_branching_points = np.empty((0, 3))
            self._branches = self.vessel_tree.branches
        self.step()

    def step(self) -> None:
        position = self.intervention.instrument_position_vessel_cs[0]
        target = self.target.coordinates
        position_branch = self.vessel_tree.find_nearest_branch_to_point(position)
        target_branch = self.vessel_tree.find_nearest_branch_to_point(target)

        (
            self._path_branching_points,
            self._path_length,
            self._path_points,
        ) = self._get_shortest_path(position_branch, target_branch, position, target)

    def _init_vessel_tree(self) -> None:
        self._node_connections = self._initialize_node_connections(
            self.vessel_tree.branching_points
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
        if len(path) == 2:
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
                    (shortest_path_points, connection.points)
                )

            target_points = target_branch.get_path_along_branch(
                path[-2].coordinates, target
            )

            target_length = get_length(target_points)
            shortest_path_points = np.vstack((shortest_path_points, target_points))
            shortest_path_length += target_length
            shortest_path = path[1:-1]

        shortest_path_points = np.unique(shortest_path_points, axis=0)

        return shortest_path, shortest_path_length, shortest_path_points

    def _create_search_graph(self, start_branch, target_branch):

        search_graph = deepcopy(self._search_graph_base)
        if start_branch == target_branch:
            search_graph["start"] = ["target"]
            return search_graph

        start_connections = []
        for branching_point in self.vessel_tree.branching_points:
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
