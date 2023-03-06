from dataclasses import dataclass
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

from .util.branch import Branch, BranchingPoint
from ..util import EveObject


@dataclass
class Insertion:
    position: np.ndarray
    direction: np.ndarray


class VesselTree(EveObject, ABC):
    # Set in subclasses in __init__() or reset():
    branches: Tuple[Branch]
    insertion: Insertion
    branching_points: List[BranchingPoint]
    coordinate_space: gym.spaces.Box = gym.spaces.Box(0.0, 0.0, (3,))

    @property
    @abstractmethod
    def mesh_path(self) -> str:
        ...

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

    def find_nearest_branch_to_point(self, point: np.ndarray) -> Branch:
        nearest_branch = None
        minDist = np.inf
        for branch in self.branches:
            distances = np.linalg.norm(branch.coordinates - point, axis=1)
            dist = np.min(distances)
            if dist < minDist:
                minDist = dist
                nearest_branch = branch
        return nearest_branch

    def at_tree_end(self, point: np.ndarray):
        branch = self.find_nearest_branch_to_point(point)
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
            for branching_point in self.branching_points:
                dist = np.linalg.norm(branching_point.coordinates - branch_point)
                if dist < branching_point.radius:
                    end_is_open = False
            return end_is_open
        else:
            return False
