from typing import List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from ....util import EveObject


@dataclass(frozen=True, eq=True)
class Branch(EveObject):
    name: str
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        if not isinstance(self.coordinates, np.ndarray):
            coordinates = np.array(self.coordinates)
            object.__setattr__(self, "coordinates", coordinates)
        _coordinates = tuple([tuple(coordinate) for coordinate in self.coordinates])
        self.coordinates.flags.writeable = False
        object.__setattr__(self, "_coordinates", _coordinates)

    def __repr__(self) -> str:
        return self.name

    @property
    def low(self) -> np.ndarray:
        return np.min(self.coordinates, axis=0)

    @property
    def high(self) -> np.ndarray:
        return np.max(self.coordinates, axis=0)

    @property
    def length(self) -> float:
        return np.sum(
            np.linalg.norm(self.coordinates[:-1] - self.coordinates[1:], axis=1)
        )

    def in_branch(self, points: np.ndarray, radius: float) -> np.ndarray:
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        broadcast_shape = [self.coordinates.shape[0]] + list(points.shape)
        points = np.broadcast_to(points, broadcast_shape)
        points = np.swapaxes(points, 0, 1)
        vectors = points - self.coordinates
        dist = np.linalg.norm(vectors, axis=-1)
        in_branch = np.any(dist < radius, axis=-1)
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
class BranchWithRadii(Branch):
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
        if not isinstance(self.coordinates, np.ndarray):
            coordinates = np.array(self.coordinates)
            object.__setattr__(self, "coordinates", coordinates)
        if not isinstance(self.radii, np.ndarray):
            radii = np.array(self.radii)
            object.__setattr__(self, "radii", radii)
        _coordinates = tuple([tuple(coordinate) for coordinate in self.coordinates])
        _radii = tuple(self.radii.tolist())
        self.coordinates.flags.writeable = False
        self.radii.flags.writeable = False
        object.__setattr__(self, "_coordinates", _coordinates)
        object.__setattr__(self, "_radii", _radii)

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

    def in_branch(self, points: np.ndarray, radius=None) -> np.ndarray:
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        broadcast_shape = [self.coordinates.shape[0]] + list(points.shape)
        points = np.broadcast_to(points, broadcast_shape)
        points = np.swapaxes(points, 0, 1)
        vectors = points - self.coordinates
        dist = np.linalg.norm(vectors, axis=-1)
        in_branch = np.any(dist < self.radii, axis=-1)
        return in_branch


@dataclass(frozen=True, eq=True)
class BranchingPoint:
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    radius: float
    connections: List[BranchWithRadii]
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        self.coordinates.flags.writeable = False
        coordinates = tuple(self.coordinates)
        object.__setattr__(self, "_coordinates", coordinates)

    def __repr__(self) -> str:
        return f"BranchingPoint({self.connections})"


def calc_branching(branches: List[Branch], radii: Union[float, List[float]]):
    raw_branching_points: List[BranchingPoint] = []
    if isinstance(radii, (float, int)):
        radii = [radii for branch in branches]

    for main_branch, main_radius in zip(branches, radii):
        # find connecting branches
        for other_branch, other_radius in zip(branches, radii):
            if other_branch == main_branch:
                continue

            points_in_main_branch = main_branch.in_branch(
                other_branch.coordinates, main_radius
            )
            if np.any(points_in_main_branch):
                idxs = np.argwhere(points_in_main_branch)
                for idx in idxs:
                    coords = other_branch.coordinates[idx[0]]
                    raw_branching_points.append(
                        BranchingPoint(
                            coords,
                            other_radius,
                            [main_branch, other_branch],
                        )
                    )

    branching_points = _consolidate_branching_points(raw_branching_points)

    return branching_points


def calc_branching_with_radii(branches: List[BranchWithRadii]):
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

    branching_points = _consolidate_branching_points(raw_branching_points)

    return branching_points


def _consolidate_branching_points(raw_branching_points: List[BranchingPoint]):
    branching_points: List[BranchingPoint] = []
    # remove duplicates from branching_point_list
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
        main_radius = np.inf
        for bp in to_average:
            coords += bp.coordinates
            main_radius = min(bp.radius, main_radius)
        coords /= len(to_average)
        branching_points.append(
            BranchingPoint(coords, main_radius, tuple(branching_point.connections))
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
                main_radius = max(branching_point.radius, other_branching_point.radius)
                connections = list(branching_point.connections) + list(
                    other_branching_point.connections
                )
                new_branching_point = BranchingPoint(
                    coord,
                    main_radius,
                    tuple(set(connections)),
                )
                raw_branching_points[i] = new_branching_point

                discard_branching_point = True

        if not discard_branching_point:
            branching_points.append(branching_point)
    return branching_points


def scale_branches_xyz(
    branches: List[Branch], xyz_scaling: Tuple[float, float, float]
) -> Tuple[Branch]:
    xyz_scaling = np.array(xyz_scaling, dtype=np.float32)
    new_branches = []
    for branch in branches:
        new_coordinates = branch.coordinates * xyz_scaling
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
    return tuple(new_branches)


def scale_branches_d(
    branches: List[BranchWithRadii], d_scaling: float
) -> Tuple[BranchWithRadii]:
    new_branches = []
    for branch in branches:
        new_radii = branch.radii * d_scaling
        new_branches.append(BranchWithRadii(branch.name, branch.coordinates, new_radii))
    return tuple(new_branches)


def scale_branches_xyzd(
    branches: List[BranchWithRadii], xyzd_scaling: Tuple[float, float, float, float]
) -> Tuple[BranchWithRadii]:
    branches = scale_branches_xyz(branches, xyzd_scaling[0:3])
    return scale_branches_d(branches, xyzd_scaling[-1])


def rotate_branches(
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
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
    return tuple(new_branches)


def omit_branches_axis(
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
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
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
