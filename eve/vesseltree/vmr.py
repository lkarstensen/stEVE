from xml.dom import minidom
from typing import List, Optional, Tuple
import os
import numpy as np
import pyvista as pv
from .vesseltree import VesselTree, Insertion, gym
from .util.branch import Branch, calc_branching, rotate
from .util import calc_insertion
from .util.meshing import get_temp_mesh_path
from .util.vmrdownload import download_vmr_files


SCALING_FACTOR = 10
LOW_HIGH_BUFFER = 3


def _get_branches(model_dir, vtu_mesh) -> List[Branch]:
    path_dir = os.path.join(model_dir, "Paths")
    files = _get_available_pths(path_dir)
    files = sorted(files)
    branches = []
    for file in files:
        file_path = os.path.join(path_dir, file)
        branch = _load_points_from_pth(file_path, vtu_mesh)
        branches.append(branch)
    return branches


def _get_available_pths(directory: str) -> List[str]:
    pth_files = []
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            pth_files.append(file)
    return pth_files


def _get_vtk_file(directory: str, file_ending: str = ".vtp") -> str:
    for file in os.listdir(directory):
        if file.endswith(file_ending):
            path = os.path.join(directory, file)
            return path


def _load_points_from_pth(pth_file_path: str, vtu_mesh: pv.UnstructuredGrid) -> Branch:
    points = []
    name = os.path.basename(pth_file_path)[:-4]
    with open(pth_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        tree = minidom.parse(file)

    xml_points = tree.getElementsByTagName("pos")

    points = []
    radii = []
    low = np.array([vtu_mesh.bounds[0], vtu_mesh.bounds[2], vtu_mesh.bounds[4]])
    high = np.array([vtu_mesh.bounds[1], vtu_mesh.bounds[3], vtu_mesh.bounds[5]])
    low += LOW_HIGH_BUFFER
    high -= LOW_HIGH_BUFFER
    for point in xml_points:
        x = float(point.attributes["x"].value) * SCALING_FACTOR
        y = float(point.attributes["y"].value) * SCALING_FACTOR
        z = float(point.attributes["z"].value) * SCALING_FACTOR
        if np.any([x, y, z] < low) or np.any([x, y, z] > high):
            continue
        points.append([x, y, z])
        radii.append(10.0)
    points = np.array(points, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)

    to_keep = vtu_mesh.find_containing_cell(points)
    to_keep += 1
    to_keep = np.argwhere(to_keep)
    points = points[to_keep.reshape(-1)]
    radii = radii[to_keep.reshape(-1)]
    return Branch(
        name=name.lower(),
        coordinates=np.array(points, dtype=np.float32),
        radii=np.array(radii, dtype=np.float32),
    )


class VMR(VesselTree):
    def __init__(
        self,
        model: str,
        insertion_point_idx: int,
        insertion_direction_idx_diff: int,
        insertion_vessel_name: str = "aorta",
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.model = model
        self.insertion_point_idx = insertion_point_idx
        self.insertion_direction_idx_diff = insertion_direction_idx_diff
        self.insertion_vessel_name = insertion_vessel_name.lower()
        self.rotate_yzx_deg = rotate_yzx_deg

        self.branches = None
        self.insertion = None
        self.branching_points = None
        self._mesh_path = None

        self._model_folder = download_vmr_files(model)
        self._mesh_folder = os.path.join(self._model_folder, "Meshes")

    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._make_mesh_obj()
        return self._mesh_path

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self.branches is None:
            self._make_branches()

    def _make_branches(self):
        mesh_path = _get_vtk_file(self._mesh_folder, ".vtu")
        mesh = pv.read(mesh_path)
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        branches = _get_branches(self._model_folder, mesh)

        if self.rotate_yzx_deg is not None:
            branches = rotate(branches, self.rotate_yzx_deg)
        self.branches = branches
        insert_vessel = self[self.insertion_vessel_name]
        ip, ip_dir = calc_insertion(
            insert_vessel,
            self.insertion_point_idx,
            self.insertion_point_idx + self.insertion_direction_idx_diff,
        )
        self.insertion = Insertion(ip, ip_dir)
        self.branching_points = calc_branching(self.branches)
        self._mesh_path = None

        branch_highs = [branch.high for branch in branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in branches]
        low = np.min(branch_lows, axis=0)
        self.coordinate_space = gym.spaces.Box(low=low, high=high)

    def _make_mesh_obj(self):
        mesh_path = _get_vtk_file(self._mesh_folder, ".vtp")
        mesh = pv.read(mesh_path)
        mesh.flip_normals()
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        if self.rotate_yzx_deg is not None:
            mesh.rotate_y(self.rotate_yzx_deg[0], inplace=True)
            mesh.rotate_z(self.rotate_yzx_deg[1], inplace=True)
            mesh.rotate_x(self.rotate_yzx_deg[2], inplace=True)
        mesh.decimate(0.9, inplace=True)

        obj_mesh_path = get_temp_mesh_path("VMR")
        pv.save_meshio(obj_mesh_path, mesh)
        self._mesh_path = obj_mesh_path
