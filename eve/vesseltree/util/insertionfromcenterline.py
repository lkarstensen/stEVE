from typing import Tuple
from .branch import Branch
import numpy as np


def calc_insertion_from_branch_start(branch: Branch) -> Tuple[np.ndarray, np.ndarray]:
    return calc_insertion(branch, 1, 2)


def calc_insertion(branch: Branch, idx_0, idx_1) -> Tuple[np.ndarray, np.ndarray]:
    point_0 = branch.coordinates[idx_0]
    point_1 = branch.coordinates[idx_1]
    insertion_point = point_0
    insertion_direction = point_1 - point_0
    insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
    return insertion_point, insertion_direction
