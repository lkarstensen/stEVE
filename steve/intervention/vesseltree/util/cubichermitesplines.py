from dataclasses import dataclass
from typing import Tuple
from typing import List
from scipy.interpolate import CubicHermiteSpline
import numpy as np


@dataclass
class CHSPoint:
    y: Tuple[float, float, float, float]
    dydx: Tuple[float, float, float, float]

    @property
    def coords(self) -> Tuple[float, float, float]:
        return (self.y[0], self.y[1], self.y[2])

    @property
    def r(self) -> float:
        return self.y[3]

    @property
    def d_coords(self) -> Tuple[float, float, float]:
        return (self.dydx[0], self.dydx[1], self.dydx[2])

    @property
    def d_r(self) -> float:
        return self.dydx[3]


def chs_point_normal(
    coords_mean: Tuple[float, float, float],
    coords_sigma: Tuple[float, float, float],
    direction_mean: Tuple[float, float, float],
    direction_sigma: Tuple[float, float, float],
    direction_magnitude_mean_and_sigma: Tuple[float, float],
    radius_mean_and_sigma: Tuple[float, float],
    d_radius_mean_and_sigma: Tuple[float, float],
    coord_offset: Tuple[float, float, float] = None,
    rng: np.random.Generator = None,
):
    if coord_offset is None:
        coord_offset = (0.0, 0.0, 0.0)
    rng = rng or np.random.default_rng()
    n = rng.normal
    y = [
        n(coords_mean[0], coords_sigma[0]) + coord_offset[0],
        n(coords_mean[1], coords_sigma[1]) + coord_offset[1],
        n(coords_mean[2], coords_sigma[2]) + coord_offset[2],
        n(radius_mean_and_sigma[0], radius_mean_and_sigma[1]),
    ]

    d_coords_direction = [
        n(d_mean, d_sigma) for d_mean, d_sigma in zip(direction_mean, direction_sigma)
    ]
    d_coords_direction = np.array(d_coords_direction, dtype=np.float32)
    magnitude = n(
        direction_magnitude_mean_and_sigma[0], direction_magnitude_mean_and_sigma[1]
    )
    d_coords = d_coords_direction / np.linalg.norm(d_coords_direction) * magnitude
    dydx = [
        d_coords[0],
        d_coords[1],
        d_coords[2],
        n(d_radius_mean_and_sigma[0], d_radius_mean_and_sigma[1]),
    ]

    return CHSPoint(y, dydx)


def chs_to_cl_points(
    points: List[CHSPoint], resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array([point.y for point in points])
    dydx = np.array([point.dydx for point in points])
    x = [0]
    for point, next_point in zip(points[:-1], points[1:]):
        vector = np.array(point.coords) - np.array(next_point.coords)
        dist = np.linalg.norm(vector)
        x.append(x[-1] + dist)

    chs = CubicHermiteSpline(x, y, dydx)
    xs = np.arange(x[0], x[-1], 0.1, dtype=np.float32)
    centerline_points = chs(xs)
    tracking_state = [centerline_points[0]]
    acc_dist = 0.0
    last_point = centerline_points[0]
    for current_point in centerline_points[1:]:

        acc_dist += np.linalg.norm(current_point[0:3] - last_point[0:3])

        if acc_dist >= resolution:
            interpolation_distance = acc_dist - resolution
            dist_between_points = np.linalg.norm(current_point[0:3] - last_point[0:3])
            unit_vector = (current_point[0:3] - last_point[0:3]) / dist_between_points
            tracking_point = current_point[0:3] - unit_vector * interpolation_distance
            diameter = (
                current_point[3]
                - (current_point[3] - last_point[3])
                / dist_between_points
                * interpolation_distance
            )
            new_point = np.append(tracking_point, diameter)
            tracking_state = np.append(tracking_state, [new_point], axis=0)
            acc_dist = interpolation_distance

        last_point = current_point

    coordinates = [point[0:3] for point in list(tracking_state)]
    coordinates = np.array(coordinates, dtype=np.float32)
    radii = [point[3] / 2 for point in list(tracking_state)]
    radii = np.array(radii, dtype=np.float32)
    return coordinates, radii
