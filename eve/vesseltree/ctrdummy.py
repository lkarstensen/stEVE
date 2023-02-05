from typing import Tuple
from . import (
    VesselTree,
    chs_to_cl_points,
    chs_point_normal,
    CHSPoint,
    CenterlineBranch,
    CenterlineTree,
)
import numpy as np


class CTRDummy(VesselTree):
    def __init__(self, seed: int = None) -> None:
        super().__init__()
        self.seed = seed
        self.create_new_geometry()

    def create_new_geometry(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        start = chs_point_normal(
            coords_mean=[0, 0, 0],
            coords_sigma=[0, 0, 0],
            direction_mean=[0, 0, 1],
            direction_sigma=[0, 0, 0],
            direction_magnitude_mean_and_sigma=[2, 0.5],
            radius_mean_and_sigma=[1, 0],
            d_radius_mean_and_sigma=[0, 0],
            rng=self.rng,
        )
        end = ctr_chs_endpoint(
            radius_azimuth_inclination_mean=[100, 0, 0],
            radius_azimuth_inclination_sigma=[20, np.pi, np.pi / 5],
            d_azimuth_inclination_sigma=[np.pi / 6, np.pi / 6],
            direction_magnitude_mean_and_sigma=[2, 0.5],
        )
        cl_points = chs_to_cl_points([start, end], 1)
        branch = CenterlineBranch(cl_points, "path")
        self.centerline_tree = CenterlineTree([branch])


def ctr_chs_endpoint(
    radius_azimuth_inclination_mean: Tuple[float, float, float],
    radius_azimuth_inclination_sigma: Tuple[float, float, float],
    d_azimuth_inclination_sigma: Tuple[float, float],
    direction_magnitude_mean_and_sigma: Tuple[float, float],
    coord_offset: Tuple[float, float, float] = None,
    rng: np.random.Generator = None,
):
    coord_offset = coord_offset or (0.0, 0.0, 0.0)
    rng = rng or np.random.default_rng()
    n = rng.normal
    radius = n(radius_azimuth_inclination_mean[0], radius_azimuth_inclination_sigma[0])
    azimuth = n(radius_azimuth_inclination_mean[1], radius_azimuth_inclination_sigma[1])
    inclination = n(
        radius_azimuth_inclination_mean[2], radius_azimuth_inclination_sigma[2]
    )

    x = radius * np.sin(inclination) * np.cos(azimuth) + coord_offset[0]
    y = radius * np.sin(inclination) * np.sin(azimuth) + coord_offset[1]
    z = radius * np.cos(inclination) + coord_offset[2]

    d_inclination_mean = 2 * inclination
    d_azimuth_mean = azimuth
    d_inclination = n(d_inclination_mean, d_azimuth_inclination_sigma[0])
    d_azimuth = n(d_azimuth_mean, d_azimuth_inclination_sigma[1])
    radius = n(
        direction_magnitude_mean_and_sigma[0], direction_magnitude_mean_and_sigma[1]
    )
    dx = radius * np.sin(d_inclination) * np.cos(d_azimuth)
    dy = radius * np.sin(d_inclination) * np.sin(d_azimuth)
    dz = radius * np.cos(d_inclination)

    return CHSPoint([x, y, z, 1], [dx, dy, dz, 0])
