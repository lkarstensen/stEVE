from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

from .util.branch import Branch, BranchingPoint
from ...util import EveObject


@dataclass
class Insertion:
    position: np.ndarray
    direction: np.ndarray


class VesselTree(EveObject, ABC):
    # Set in subclasses in __init__() or reset():
    branches: Tuple[Branch]
    branching_points: List[BranchingPoint]
    centerline_coordinates: np.ndarray
    insertion: Insertion
    coordinate_space: gym.spaces.Box
    coordinate_space_episode: gym.spaces.Box
    mesh_path: str
    visu_mesh_path: Optional[str] = None

    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr=0, seed: int = None) -> None:
        ...

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, int):
            idx = item
        else:
            branch_names = tuple(branch.name for branch in self.branches)
            idx = branch_names.index(item)
        return self.branches[idx]

    def values(self) -> Tuple[Branch]:
        return self.branches

    def keys(self) -> Tuple[str]:
        return tuple(branch.name for branch in self.branches)

    def items(self):
        branch_names = tuple(branch.name for branch in self.branches)
        return zip(branch_names, self.branches)

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "branches": self.branches,
            "branching_points": self.branching_points,
            "centerline_coordinates": self.centerline_coordinates,
            "insertion": self.insertion,
            "coordinate_space": self.coordinate_space,
            "coordinate_space_episode": self.coordinate_space_episode,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {}
        return state


def find_nearest_branch_to_point(
    point: np.ndarray,
    vessel_tree: VesselTree,
) -> Branch:
    nearest_branch = None
    min_dist = np.inf
    for branch in vessel_tree.branches:
        distances = np.linalg.norm(branch.coordinates - point, axis=1)
        dist = np.min(distances)
        if dist < min_dist:
            min_dist = dist
            nearest_branch = branch
    return nearest_branch


def at_tree_end(
    point: np.ndarray,
    vessel_tree: VesselTree,
) -> bool:
    if vessel_tree.branches is None:
        return False
    branch = find_nearest_branch_to_point(point, vessel_tree)
    branch_np = branch.coordinates
    distances = np.linalg.norm(branch_np - point, axis=1)
    min_idx = np.argmin(distances)
    sec_min_idx = np.argpartition(distances, 1)[1]
    min_to_sec_min = branch_np[sec_min_idx] - branch_np[min_idx]
    min_to_point = point - branch_np[min_idx]
    dot_prod = np.dot(min_to_sec_min, min_to_point)

    if (min_idx == 0 or min_idx == branch_np.shape[0] - 1) and dot_prod <= 0:
        branch_point = branch.coordinates[min_idx]
        end_is_open = True
        for branching_point in vessel_tree.branching_points:
            dist = np.linalg.norm(branching_point.coordinates - branch_point)
            if dist < branching_point.radius:
                end_is_open = False
        return end_is_open
    else:
        return False


def plot_branches(vesseltree: VesselTree):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    coord_space = vesseltree.coordinate_space_episode
    margins = [
        (coord_space.high[0] - coord_space.low[0]),
        (coord_space.high[1] - coord_space.low[1]),
        (coord_space.high[2] - coord_space.low[2]),
    ]
    margin = max(margins)
    ax.set_xlim3d(
        coord_space.low[0],
        coord_space.low[0] + margin,
    )
    ax.set_ylim3d(
        coord_space.low[1],
        coord_space.low[1] + margin,
    )
    ax.set_zlim3d(
        coord_space.low[2],
        coord_space.low[2] + margin,
    )
    pick_event_handler = PickEventHandler(vesseltree)
    fig.canvas.mpl_connect("pick_event", pick_event_handler.on_click)
    for branch in vesseltree.branches:
        x = np.delete(branch.coordinates, [1, 2], axis=1).reshape(
            -1,
        )
        y = np.delete(branch.coordinates, [0, 2], axis=1).reshape(
            -1,
        )
        z = np.delete(branch.coordinates, [0, 1], axis=1).reshape(
            -1,
        )
        # ax.plot3D(x, y, z)
        line = mplot3d.art3d.Line3D(
            x, y, z, picker=True, pickradius=3, label=branch.name, color="red"
        )
        ax.add_artist(line)

    x = [
        branching_point.coordinates[0]
        for branching_point in vesseltree.branching_points
    ]
    y = [
        branching_point.coordinates[1]
        for branching_point in vesseltree.branching_points
    ]
    z = [
        branching_point.coordinates[2]
        for branching_point in vesseltree.branching_points
    ]

    bp_artist = mplot3d.art3d.Line3D(
        x, y, z, marker="o", color="g", markersize=8, linestyle="None"
    )
    ax.add_artist(bp_artist)
    # fig.canvas.draw()
    # plt.pause(0.001)
    # fig.canvas.start_event_loop(0.00001)
    plt.show()


class PickEventHandler:
    def __init__(self, vessel_tree: VesselTree) -> None:
        self.vessel_tree = vessel_tree

    def on_click(self, event):
        data = event.artist.get_data_3d()
        idx = event.ind[0] + int((event.ind[-1] - event.ind[0]) / 2)
        x = data[0][idx]
        y = data[1][idx]
        z = data[2][idx]
        point = np.array([x, y, z])
        branch = find_nearest_branch_to_point(point, self.vessel_tree)

        dist_to_point = np.linalg.norm(branch.coordinates - point, axis=1)
        idx = np.argmin(dist_to_point)

        print(branch.name)
        print(point)
        print(f"{idx=}")
