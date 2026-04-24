from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

from ...vesseltree.util.meshing import get_temp_mesh_path


@dataclass
class StraightMeshElement:
    length: float


@dataclass
class ArcMeshElement:
    radius: float
    angle_in_plane_deg: float
    angle_out_of_plane_deg: float
    resolution: float


def _deg_to_rad(deg: float) -> float:
    return deg * np.pi / 180


def save_line_mesh_to_file(point_cloud: np.ndarray, file: str):
    with open(file, "w", encoding="utf-8") as f:
        vertices = [
            f"v {point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n" for point in point_cloud
        ]
        f.writelines(vertices)
        connections = [f"l {i+1} {i+2}\n" for i in range(point_cloud.shape[0] - 1)]
        f.writelines(connections)


def save_line_mesh(instrument_point_cloud: np.ndarray) -> str:
    mesh_path = get_temp_mesh_path("endovascular_instrument")
    save_line_mesh_to_file(instrument_point_cloud, mesh_path)
    return mesh_path


def load_mesh(file_path: str) -> np.ndarray:
    point_cloud = []
    with open(file_path, "w", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("v"):
                coords = line[2:]
                coords = [float(axis) for axis in coords.split(" ")]
                point_cloud.append(coords)
            else:
                break
    return np.array(point_cloud)


def _get_combined_angle_axis(
    in_plane_angle: float,
    out_of_plane_angle: float,
    in_plane_axis: np.ndarray,
    out_of_plane_axis: np.ndarray,
):
    axis = (in_plane_axis * in_plane_angle + out_of_plane_axis * out_of_plane_angle) / (
        abs(in_plane_angle) + abs(out_of_plane_angle)
    )
    angle = (in_plane_angle**2 + out_of_plane_angle**2) / (
        abs(in_plane_angle) + abs(out_of_plane_angle)
    )

    return angle, axis


def _rotate_around_axis(vector: np.ndarray, angle: float, axis: np.ndarray):
    axis = axis / np.linalg.norm(axis)
    x, y, z = tuple(axis)
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array(
        [
            [
                cos + x**2 * (1 - cos),
                x * y * (1 - cos) - z * sin,
                x * z * (1 - cos) + y * sin,
            ],
            [
                y * x * (1 - cos) + z * sin,
                cos + y**2 * (1 - cos),
                y * z * (1 - cos) - x * sin,
            ],
            [
                z * x * (1 - cos) - y * sin,
                z * y * (1 - cos) + x * sin,
                cos + z**2 * (1 - cos),
            ],
        ]
    )

    return np.matmul(R, vector)


def _add_straight_part(
    straight_element: StraightMeshElement,
    last_point: np.ndarray,
    direction: np.ndarray,
    point_clouds: List[np.ndarray],
) -> None:
    length = straight_element.length
    start = last_point

    sample_points: np.ndarray = np.linspace(0.0, length, 2, endpoint=True)
    sample_points = sample_points[1:]
    shape = (sample_points.shape[0], 3)
    point_cloud = np.full(shape, direction)
    point_cloud *= sample_points[:, None]
    point_cloud += start
    point_cloud = np.round(point_cloud, 4)

    last_point = point_cloud[-1]
    point_clouds.append(point_cloud)
    return last_point, direction, point_clouds


def _add_curve_part(
    arc_def: ArcMeshElement,
    last_point: np.ndarray,
    direction: np.ndarray,
    in_plane_axis: np.ndarray,
    out_of_plane_axis: np.ndarray,
    point_clouds: List[np.ndarray],
) -> None:
    start = last_point
    initial_direction = direction
    angle_in_plane = _deg_to_rad(arc_def.angle_in_plane_deg)
    angle_out_of_plane = _deg_to_rad(arc_def.angle_out_of_plane_deg)
    radius = arc_def.radius
    resolution = arc_def.resolution

    angle, axis = _get_combined_angle_axis(
        angle_in_plane, angle_out_of_plane, in_plane_axis, out_of_plane_axis
    )

    dir_to_curve_center = _rotate_around_axis(initial_direction, np.pi / 2, axis)
    curve_center = start + dir_to_curve_center * radius

    arc_length = radius * abs(angle)
    n_points = int(np.ceil(arc_length / resolution)) + 1
    sample_angles = np.linspace(0.0, angle, n_points, endpoint=True)
    sample_angles = sample_angles[1:]

    base_vector = -dir_to_curve_center * radius
    vectors = [_rotate_around_axis(base_vector, angle, axis) for angle in sample_angles]
    vectors = np.array(vectors)

    curve_point_cloud = vectors + curve_center
    curve_point_cloud = np.round(curve_point_cloud, 4)
    direction = _rotate_around_axis(initial_direction, angle, axis)
    out_of_plane_axis = _rotate_around_axis(out_of_plane_axis, angle, axis)
    in_plane_axis = _rotate_around_axis(in_plane_axis, angle, axis)
    last_point = curve_point_cloud[-1]
    point_clouds.append(curve_point_cloud)

    return last_point, direction, in_plane_axis, out_of_plane_axis, point_clouds


def create_shape_point_cloud(
    elements: List[Union[StraightMeshElement, ArcMeshElement]],
) -> Tuple[np.ndarray, float]:
    in_plane_axis = np.array([0, 0, 1])
    out_of_plane_axis = np.array([0, 1, 0])

    last_point = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    point_clouds = [last_point.reshape(1, -1)]
    for element in elements:
        if isinstance(element, ArcMeshElement):
            (
                last_point,
                direction,
                in_plane_axis,
                out_of_plane_axis,
                point_clouds,
            ) = _add_curve_part(
                element,
                last_point,
                direction,
                in_plane_axis,
                out_of_plane_axis,
                point_clouds,
            )
        elif isinstance(element, StraightMeshElement):
            last_point, direction, point_clouds = _add_straight_part(
                element, last_point, direction, point_clouds
            )

    point_cloud = np.concatenate(point_clouds, axis=0)
    length = calc_length_of_line_point_cloud(point_cloud)

    return point_cloud, length


def calc_length_of_line_point_cloud(point_cloud: np.ndarray) -> float:
    pc_diff = point_cloud[:-1] - point_cloud[1:]
    length = np.sum(np.linalg.norm(pc_diff, axis=-1)).tolist()
    return length
