from itertools import product
from typing import List, Optional
import random
import numpy as np
import gymnasium as gym

from . import VesselTree, AorticArch, ArchType


class AorticArchRandom(VesselTree):
    def __init__(
        self,
        seed_random: Optional[int] = None,
        scale_width_array: List[float] = np.linspace(0.7, 1.3, 200, endpoint=True),
        scale_heigth_array: List[float] = np.linspace(0.7, 1.3, 200, endpoint=True),
        scale_diameter_array: List[float] = np.linspace(0.7, 1.3, 200, endpoint=True),
        arch_types_filter: Optional[List[ArchType]] = None,
        seeds_vessel: Optional[List[int]] = None,
        rotate_y_deg_array: Optional[List[float]] = None,
        rotate_z_deg_array: Optional[List[float]] = None,
        rotate_x_deg_array: Optional[List[float]] = None,
        omit_axis: Optional[str] = None,
        n_coordinate_space_iters: int = 5,
        episodes_between_change: int = 1,
    ) -> None:
        self.seed_random = seed_random
        self.scale_width_array = scale_width_array
        self.scale_heigth_array = scale_heigth_array
        self.scale_diameter_array = scale_diameter_array
        self.seeds_vessel = seeds_vessel or [None]
        self.arch_types_filter = arch_types_filter
        self.rotate_y_deg_array = rotate_y_deg_array or [0.0]
        self.rotate_z_deg_array = rotate_z_deg_array or [0.0]
        self.rotate_x_deg_array = rotate_x_deg_array or [0.0]
        self.omit_axis = omit_axis
        self.n_coordinate_space_iters = n_coordinate_space_iters
        self.episodes_between_change = episodes_between_change

        all_archtypes = tuple(arch for arch in ArchType)

        if arch_types_filter is not None:
            self._arch_types = tuple(set(all_archtypes) & set(arch_types_filter))
        else:
            self._arch_types = all_archtypes

        self._rng = random.Random(seed_random)
        self._vessel_tree: AorticArch = None

    @property
    def mesh_path(self) -> str:
        return self._vessel_tree.mesh_path

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        if self._vessel_tree is None:
            self._calc_coordinate_space(self.n_coordinate_space_iters)
            self._vessel_tree = self._randomize_vessel()
        elif episode_nr % self.episodes_between_change == 0:
            self._vessel_tree = self._randomize_vessel()

        self._vessel_tree.reset(episode_nr, seed)
        self.branches = self._vessel_tree.branches
        self.insertion = self._vessel_tree.insertion
        self.branching_points = self._vessel_tree.branching_points

    def _calc_coordinate_space(self, iterations) -> None:
        width_high = np.max(self.scale_width_array)
        heigth_high = np.max(self.scale_heigth_array)
        diameter_high = np.max(self.scale_diameter_array)
        rot_x_low_high = {
            np.min(self.rotate_x_deg_array),
            np.max(self.rotate_x_deg_array),
        }
        rot_y_low_high = {
            np.min(self.rotate_y_deg_array),
            np.max(self.rotate_y_deg_array),
        }
        rot_z_low_high = {
            np.min(self.rotate_z_deg_array),
            np.max(self.rotate_z_deg_array),
        }

        low_global = np.array([np.inf, np.inf, np.inf])
        high_global = np.array([-np.inf, -np.inf, -np.inf])

        combinations = product(
            self._arch_types, rot_x_low_high, rot_y_low_high, rot_z_low_high
        )
        for _ in range(iterations):
            for archtype, rot_x, rot_y, rot_z in combinations:
                vessel_seed = self._rng.choice(self.seeds_vessel)
                vessel = AorticArch(
                    archtype,
                    vessel_seed,
                    [rot_y, rot_z, rot_x],
                    [width_high, width_high, heigth_high, diameter_high],
                    self.omit_axis,
                )
                vessel.reset()
                low_global = np.minimum(low_global, vessel.coordinate_space.low)
                high_global = np.maximum(high_global, vessel.coordinate_space.high)
        self.coordinate_space = gym.spaces.Box(low=low_global, high=high_global)

    def _randomize_vessel(self):
        arch_type = self._rng.choice(self._arch_types)
        vessel_seed = self._rng.choice(self.seeds_vessel)
        xy_scaling = self._rng.choice(self.scale_width_array)
        z_scaling = self._rng.choice(self.scale_heigth_array)
        diameter_scaling = self._rng.choice(self.scale_diameter_array)
        rot_x = self._rng.choice(self.rotate_x_deg_array)
        rot_y = self._rng.choice(self.rotate_y_deg_array)
        rot_z = self._rng.choice(self.rotate_z_deg_array)
        return AorticArch(
            arch_type,
            vessel_seed,
            [rot_y, rot_z, rot_x],
            [xy_scaling, xy_scaling, z_scaling, diameter_scaling],
            self.omit_axis,
        )
