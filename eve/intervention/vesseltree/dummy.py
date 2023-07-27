from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym

from .vesseltree import VesselTree, Insertion
from .util.branch import Branch, calc_branching


class VesselTreeDummy(VesselTree):
    def __init__(
        self,
        branch_centerlines: Union[List[np.ndarray], Dict[str, np.ndarray]],
        radii_aprox: Optional[Union[List[float], float]] = None,
        insertion_point: Optional[Tuple[float, float, float]] = None,
        insertion_direction: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.branch_centerlines = branch_centerlines
        self.radii_aprox = radii_aprox
        self.insertion_point = insertion_point
        self.insertion_direction = insertion_direction

        if isinstance(branch_centerlines, list):
            branche_new = [
                Branch(f"branch_{i}", coords)
                for i, coords in enumerate(branch_centerlines)
            ]
        elif isinstance(branch_centerlines, dict):
            branche_new = [
                Branch(name, coords) for name, coords in branch_centerlines.items()
            ]
        self.branches = branche_new

        branch_cl = [branch.coordinates for branch in self.branches]
        self.centerline_coordinates = np.concatenate(branch_cl)

        self.branching_points = (
            calc_branching(self.branches, radii_aprox)
            if radii_aprox is not None
            else None
        )

        self.insertion = Insertion(
            np.array(insertion_point), np.array(insertion_direction)
        )

        branch_highs = [branch.high for branch in self.branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in self.branches]
        low = np.min(branch_lows, axis=0)
        coord_space = gym.spaces.Box(low=low, high=high)
        self.coordinate_space = coord_space
        self.coordinate_space_episode = coord_space

        self.mesh_path = None
        self.visu_mesh_path = None

    def step(self) -> None:
        ...

    def reset(self, episode_nr=0, seed: int = None) -> None:
        ...
