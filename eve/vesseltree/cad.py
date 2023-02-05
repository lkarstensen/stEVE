from typing import List, Optional, Tuple
import os
import csv
import numpy as np

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import Branch, calc_branching, scale, rotate
from .util import calc_insertion_from_branch_start
from .util.meshing import generate_temp_mesh


def _get_branches(data_dir):
    files = _get_available_csvs(data_dir, "branch")
    files = sorted(files)
    branches = []
    for file in files:
        branch = _load_points_from_csv(data_dir + "/" + file)
        branches.append(branch)
    return branches


def _get_available_csvs(directory: str, prefix: str) -> List[str]:
    csv_files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(".csv"):
            csv_files.append(file)
    return csv_files


def _load_points_from_csv(csv_path: str) -> Branch:
    points = []
    name = os.path.basename(csv_path)[:-4]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        _ = next(reader)
        points = []
        radii = []
        for row in reader:
            point = [
                float(row[0].replace(",", ".")),
                float(row[1].replace(",", ".")),
                float(row[2].replace(",", ".")),
            ]
            radius = float(row[3].replace(",", "."))
            points.append(point)
            radii.append(radius)

    return Branch(name=name, coordinates=np.array(points), radii=np.array(radii))


class OrganicV2(VesselTree):
    def __init__(
        self,
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,
        scale_xyzd: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.rotate_yzx_deg = rotate_yzx_deg
        self.scale_xyzd = scale_xyzd

        self._mesh_path = None
        self.branches = None
        self._data_name = "organicv2"

    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._mesh_path = generate_temp_mesh(self.branches, "cad_vessel_tree", 0.99)
        return self._mesh_path

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self.branches is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_directory, "data", "cad", self._data_name)
            branches = _get_branches(data_dir)
            if self.rotate_yzx_deg is not None:
                branches = rotate(branches, self.rotate_yzx_deg)
            if self.scale_xyzd is not None:
                branches = scale(branches, self.scale_xyzd)

            insertion_point, ip_dir = calc_insertion_from_branch_start(branches[0])

            branch_highs = [branch.high for branch in branches]
            high = np.max(branch_highs, axis=0)
            branch_lows = [branch.low for branch in branches]
            low = np.min(branch_lows, axis=0)

            self.branches = branches
            self.insertion = Insertion(insertion_point, ip_dir)
            self.coordinate_space = gym.spaces.Box(low=low, high=high)
            self.branching_points = calc_branching(branches)
            self._mesh_path = None


class Organic3DV2(OrganicV2):
    def __init__(
        self,
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,
        scale_xyzd: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        super().__init__(rotate_yzx_deg, scale_xyzd)
        self._data_name = "organic3dv2"
