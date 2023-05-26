from typing import Tuple, List

from ..util.branch import BranchWithRadii
from ..util.cubichermitesplines import CHSPoint, chs_point_normal, chs_to_cl_points

import numpy as np


def brachiocephalic_trunk_static(
    start: Tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator = None,
) -> Tuple[BranchWithRadii, List[CHSPoint]]:
    rng = rng or np.random.default_rng()
    chs_points: List[CHSPoint] = []
    chs_points.append(
        chs_point_normal(
            coords_mean=(0.0, 0.0, 0.0),
            coords_sigma=(0.0, 0.0, 0.0),
            direction_mean=(0, 0.1, 1),
            direction_sigma=(0.1, 0.1, 0.0),
            direction_magnitude_mean_and_sigma=(2.0, 0.1),
            radius_mean_and_sigma=(22.0, 1.0),
            d_radius_mean_and_sigma=(0.0, 0.0),
            coord_offset=start,
            rng=rng,
        )
    )
    chs_points.append(
        chs_point_normal(
            coords_mean=(-20.0, 5.0, 45.0),
            coords_sigma=(3.0, 2.0, 3.0),
            direction_mean=(-0.3, 0.3, 0.7),
            direction_sigma=(0.1, 0.1, 0.3),
            direction_magnitude_mean_and_sigma=(0.8, 0.1),
            radius_mean_and_sigma=(chs_points[-1].r - 10, 0.7),
            d_radius_mean_and_sigma=(-0.1, 0.033),
            coord_offset=start,
            rng=rng,
        )
    )
    cl_coordinates, cl_radii = chs_to_cl_points(chs_points, resolution)
    return BranchWithRadii("bct", cl_coordinates, cl_radii), chs_points
