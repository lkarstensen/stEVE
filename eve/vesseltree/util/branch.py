from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, eq=True)
class Branch:
    name: str
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    radii: np.ndarray = field(init=True, compare=False, repr=True)
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )
    _radii: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        coordinates = tuple([tuple(coordinate) for coordinate in self.coordinates])
        radii = tuple(self.radii.tolist())
        self.coordinates.flags.writeable = False
        self.radii.flags.writeable = False
        object.__setattr__(self, "_coordinates", coordinates)
        object.__setattr__(self, "_radii", radii)

    def __repr__(self) -> str:
        return self.name

    @property
    def low(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_low = self.coordinates - radii
        return np.min(coords_low, axis=0)

    @property
    def high(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_high = self.coordinates + radii
        return np.max(coords_high, axis=0)

    @property
    def length(self) -> float:
        return np.sum(
            np.linalg.norm(self.coordinates[:-1] - self.coordinates[1:], axis=1)
        )

    def in_branch(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        broadcast_shape = [self.coordinates.shape[0]] + list(points.shape)
        points = np.broadcast_to(points, broadcast_shape)
        points = np.swapaxes(points, 0, 1)
        vectors = points - self.coordinates
        dist = np.linalg.norm(vectors, axis=-1)
        in_branch = np.any(dist < self.radii, axis=-1)
        return in_branch

    def get_path_along_branch(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        start_to_branch_dist = np.linalg.norm(self.coordinates - start, axis=1)
        start_idx = np.argmin(start_to_branch_dist)
        end_to_branch_dist = np.linalg.norm(self.coordinates - end, axis=1)
        end_idx = np.argmin(end_to_branch_dist)
        idx_diff = end_idx - start_idx
        if abs(idx_diff) == 0:
            idx_dir = 1
            start_idx += 1
            end_idx -= 1
        else:
            # use next idx to prevent hopping between points
            idx_dir = int(idx_diff / abs(idx_diff))
            start_idx += idx_dir
            end_idx -= idx_dir

        partial_branch = self.coordinates[start_idx : end_idx + idx_dir : idx_dir]
        path = np.concatenate(
            [start.reshape(1, 3), partial_branch, end.reshape(1, 3)], axis=0
        )
        return path


@dataclass(frozen=True, eq=True)
class BranchingPoint:
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    radius: float
    connections: List[Branch]
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        self.coordinates.flags.writeable = False
        coordinates = tuple(self.coordinates)
        object.__setattr__(self, "_coordinates", coordinates)

    def __repr__(self) -> str:
        return f"BranchingPoint({self.connections})"


def calc_branching(branches: Tuple[Branch]):
    raw_branching_points: List[BranchingPoint] = []

    for main_branch in branches:
        # find connecting branches
        for other_branch in branches:
            if other_branch == main_branch:
                continue

            points_in_main_branch = main_branch.in_branch(other_branch.coordinates)
            if np.any(points_in_main_branch):
                idxs = np.argwhere(points_in_main_branch)
                for idx in idxs:
                    coords = other_branch.coordinates[idx[0]]
                    radius = other_branch.radii[idx[0]]
                    raw_branching_points.append(
                        BranchingPoint(
                            coords,
                            radius,
                            [main_branch, other_branch],
                        )
                    )

    # remove duplicates from branching_point_list
    branching_points: List[BranchingPoint] = []
    while raw_branching_points:
        branching_point = raw_branching_points.pop(-1)
        to_average = [branching_point]
        for i, other_branching_point in enumerate(raw_branching_points):
            if set(other_branching_point.connections) == set(
                branching_point.connections
            ):
                # add connections of branching point to other branching point
                to_average.append(other_branching_point)

        for bp in to_average[1:]:
            raw_branching_points.remove(bp)
        coords = np.array([0.0, 0.0, 0.0])
        radius = np.inf
        for bp in to_average:
            coords += bp.coordinates
            radius = min(bp.radius, radius)
        coords /= len(to_average)
        branching_points.append(
            BranchingPoint(coords, radius, tuple(branching_point.connections))
        )

    raw_branching_points = branching_points
    branching_points: List[BranchingPoint] = []

    while raw_branching_points:
        branching_point = raw_branching_points.pop(-1)
        discard_branching_point = False
        for i, other_branching_point in enumerate(raw_branching_points):
            distance = np.linalg.norm(
                branching_point.coordinates - other_branching_point.coordinates
            )
            check_distance = (
                distance < branching_point.radius + other_branching_point.radius
            )

            if check_distance:
                # add connections of branching point to other branching point
                coord = branching_point.coordinates + other_branching_point.coordinates
                coord /= 2
                radius = max(branching_point.radius, other_branching_point.radius)
                connections = list(branching_point.connections) + list(
                    other_branching_point.connections
                )
                new_branching_point = BranchingPoint(
                    coord,
                    radius,
                    tuple(set(connections)),
                )
                raw_branching_points[i] = new_branching_point

                discard_branching_point = True

        if not discard_branching_point:
            branching_points.append(branching_point)

    return branching_points


def scale(
    branches: List[Branch], scale_xyzd: Tuple[float, float, float, float]
) -> Tuple[Branch]:
    xyz_sclaing = np.array(scale_xyzd, dtype=np.float32)[:-1]
    new_branches = []
    for branch in branches:
        new_radii = branch.radii * scale_xyzd[-1]
        new_coordinates = branch.coordinates * xyz_sclaing
        new_branches.append(Branch(branch.name, new_coordinates, new_radii))

    return tuple(new_branches)


def rotate(
    branches: List[Branch], rotate_yzx_deg: Tuple[float, float, float]
) -> Tuple[Branch]:
    new_branches = []
    for branch in branches:
        new_coordinates = _rotate_array(
            array=branch.coordinates,
            y_deg=rotate_yzx_deg[0],
            z_deg=rotate_yzx_deg[1],
            x_deg=rotate_yzx_deg[2],
        )
        new_branches.append(Branch(branch.name, new_coordinates, branch.radii))
    return tuple(new_branches)


def fill_axis_with_dummy_value(
    branches: List[Branch], axis_to_remove: str, dummy_value: float = 0
) -> Tuple[Branch]:
    if axis_to_remove not in ["x", "y", "z"]:
        raise ValueError(f"to_2d() {axis_to_remove =} has to be 'x', 'y' or 'z'")
    convert = {"x": 0, "y": 1, "z": 2}
    axis_to_remove = convert[axis_to_remove]
    new_branches = []
    for branch in branches:
        new_coordinates = np.delete(branch.coordinates, axis_to_remove, axis=1)
        new_coordinates = np.insert(
            new_coordinates, axis_to_remove, dummy_value, axis=1
        )
        new_branches.append(Branch(branch.name, new_coordinates, branch.radii))
    return tuple(new_branches)


def _rotate_array(
    array: np.ndarray,
    y_deg: float,
    z_deg: float,
    x_deg: float,
):
    y_rad = y_deg * np.pi / 180
    lao_rao_rad = z_deg * np.pi / 180
    cra_cau_rad = x_deg * np.pi / 180

    rotation_matrix_y = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)],
        ],
        dtype=np.float32,
    )

    rotation_matrix_lao_rao = np.array(
        [
            [np.cos(lao_rao_rad), -np.sin(lao_rao_rad), 0],
            [np.sin(lao_rao_rad), np.cos(lao_rao_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    rotation_matrix_cra_cau = np.array(
        [
            [1, 0, 0],
            [0, np.cos(cra_cau_rad), -np.sin(cra_cau_rad)],
            [0, np.sin(cra_cau_rad), np.cos(cra_cau_rad)],
        ],
        dtype=np.float32,
    )
    rotation_matrix = np.matmul(rotation_matrix_cra_cau, rotation_matrix_lao_rao)
    rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_y)
    # transpose such that matrix multiplication works
    rotated_array = np.matmul(rotation_matrix, array.T).T
    return rotated_array
