from dataclasses import dataclass, field
from typing import List, Tuple, Union
import numpy as np
from .device3d import NonProceduralShape
from .linemeshgenerator import save_line_mesh
from ...vesseltree.util.meshing import get_temp_mesh_path


def deg_to_rad(deg: float) -> float:
    return deg * np.pi / 180


@dataclass(frozen=True)
class StraightPart:
    length: float
    visu_edges_per_mm: float
    collis_edges_per_mm: float
    beams_per_mm: float


@dataclass(frozen=True)
class Arc:
    radius: float
    angle_in_plane_deg: float
    angle_out_of_plane_deg: float
    visu_edges_per_mm: float
    collis_edges_per_mm: float
    beams_per_mm: float
    resolution: float = 0.1


@dataclass(frozen=True)
class MeshDevice(NonProceduralShape):
    name: str
    velocity_limit: Tuple[float, float]
    elements: List[Union[StraightPart, Arc]]
    outer_diameter: float
    inner_diameter: float
    poisson_ratio: float
    young_modulus: float
    mass_density: float
    color: Tuple[float, float, float]

    mesh_path: str = field(repr=False, init=False, compare=False, default=None)
    length: float = field(repr=False, init=False, compare=False, default=0.0)
    radius: float = field(repr=False, init=False, compare=False, default=0.0)
    inner_radius: float = field(repr=False, init=False, compare=False, default=0.0)
    num_edges: int = field(repr=False, init=False, compare=False, default=0)
    num_edges_collis: int = field(repr=False, init=False, compare=False, default=0)
    density_of_beams: int = field(repr=False, init=False, compare=False, default=0)
    key_points: List[float] = field(
        repr=False, init=False, compare=False, default_factory=list
    )

    def __post_init__(self):

        (
            point_cloud,
            key_points,
            visu_edges,
            collis_edges,
            beams,
        ) = self._create_shape_point_cloud()
        mesh_path = self._create_mesh(point_cloud)

        radius = self.outer_diameter / 2
        inner_radius = self.inner_diameter / 2
        length = key_points[-1]
        object.__setattr__(self, "mesh_path", mesh_path)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "inner_radius", inner_radius)
        object.__setattr__(self, "num_edges", visu_edges)
        object.__setattr__(self, "num_edges_collis", collis_edges)
        object.__setattr__(self, "density_of_beams", beams)
        object.__setattr__(self, "key_points", key_points)
        object.__setattr__(self, "elements", tuple(self.elements))

    def _create_shape_point_cloud(self) -> np.ndarray:

        in_plane_axis = np.array([0, 0, 1])
        out_of_plane_axis = np.array([0, 1, 0])

        last_point = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        key_points = [0.0]
        visu_edges = []
        collis_edges = []
        beams = []
        point_clouds = [last_point.reshape(1, -1)]

        for element in self.elements:
            if isinstance(element, Arc):
                (
                    last_point,
                    direction,
                    in_plane_axis,
                    out_of_plane_axis,
                    point_clouds,
                ) = self._add_curve_part(
                    element,
                    last_point,
                    direction,
                    in_plane_axis,
                    out_of_plane_axis,
                    point_clouds,
                )
                pc_diff = point_clouds[-1][:-1] - point_clouds[-1][1:]
                lengths = np.linalg.norm(pc_diff, axis=-1)
                length = np.sum(lengths)
            elif isinstance(element, StraightPart):
                last_point, direction, point_clouds = self._add_straight_part(
                    element, last_point, direction, point_clouds
                )
                length = element.length
            key_points.append(key_points[-1] + length)
            visu_edges.append(int(np.ceil(length * element.visu_edges_per_mm)))
            collis_edges.append(int(np.ceil(length * element.collis_edges_per_mm)))
            beams.append(int(np.ceil(length * element.beams_per_mm)))

        point_cloud = np.concatenate(point_clouds, axis=0)

        return (
            point_cloud,
            tuple(key_points),
            tuple(visu_edges),
            tuple(collis_edges),
            tuple(beams),
        )

    def _add_straight_part(
        self,
        straight_element: StraightPart,
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
        self,
        arc_def: Arc,
        last_point: np.ndarray,
        direction: np.ndarray,
        in_plane_axis: np.ndarray,
        out_of_plane_axis: np.ndarray,
        point_clouds: List[np.ndarray],
    ) -> None:
        start = last_point
        initial_direction = direction
        angle_in_plane = deg_to_rad(arc_def.angle_in_plane_deg)
        angle_out_of_plane = deg_to_rad(arc_def.angle_out_of_plane_deg)
        radius = arc_def.radius
        resolution = arc_def.resolution

        angle, axis = self._get_combined_angle_axis(
            angle_in_plane, angle_out_of_plane, in_plane_axis, out_of_plane_axis
        )

        dir_to_curve_center = self._rotate_around_axis(
            initial_direction, np.pi / 2, axis
        )
        curve_center = start + dir_to_curve_center * radius

        arc_length = radius * abs(angle)
        n_points = int(np.ceil(arc_length / resolution)) + 1
        sample_angles = np.linspace(0.0, angle, n_points, endpoint=True)
        sample_angles = sample_angles[1:]

        base_vector = -dir_to_curve_center * radius
        vectors = [
            self._rotate_around_axis(base_vector, angle, axis)
            for angle in sample_angles
        ]
        vectors = np.array(vectors)

        curve_point_cloud = vectors + curve_center
        curve_point_cloud = np.round(curve_point_cloud, 4)
        direction = self._rotate_around_axis(initial_direction, angle, axis)
        out_of_plane_axis = self._rotate_around_axis(out_of_plane_axis, angle, axis)
        in_plane_axis = self._rotate_around_axis(in_plane_axis, angle, axis)
        last_point = curve_point_cloud[-1]
        point_clouds.append(curve_point_cloud)

        return last_point, direction, in_plane_axis, out_of_plane_axis, point_clouds

    def _get_combined_angle_axis(
        self,
        in_plane_angle: float,
        out_of_plane_angle: float,
        in_plane_axis: np.ndarray,
        out_of_plane_axis: np.ndarray,
    ):
        axis = (
            in_plane_axis * in_plane_angle + out_of_plane_axis * out_of_plane_angle
        ) / (abs(in_plane_angle) + abs(out_of_plane_angle))
        angle = (in_plane_angle**2 + out_of_plane_angle**2) / (
            abs(in_plane_angle) + abs(out_of_plane_angle)
        )

        return angle, axis

    @staticmethod
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

    def _create_mesh(self, device_point_cloud: np.ndarray) -> str:
        mesh_path = get_temp_mesh_path("endovascular_instrument")
        save_line_mesh(device_point_cloud, mesh_path)
        return mesh_path
