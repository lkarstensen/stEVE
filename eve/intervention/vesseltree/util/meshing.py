from tempfile import gettempdir
import os
from typing import Iterable, Tuple
import numpy as np
import pyvista as pv
from .branch import BranchWithRadii
from .voxelcube import VoxelCube, create_empty_voxel_cube_from_branches


def get_surface_mesh(
    voxel_cube: VoxelCube, gradient_direction: str = "descent", level: float = None
) -> pv.PolyData:
    from skimage import measure

    vertices, faces, _, _ = measure.marching_cubes(
        voxel_cube.value_array,
        level,
        spacing=voxel_cube.spacing,
        gradient_direction=gradient_direction,
    )
    faces_nr_points = np.ones((faces.shape[0], 1), dtype=np.int64)
    faces_nr_points *= 3
    faces = np.concatenate((faces_nr_points, faces), axis=1)
    mesh = pv.PolyData(vertices, faces)
    mesh.translate(voxel_cube.world_offset, inplace=True)
    return mesh


def load_mesh(path: str) -> pv.PolyData:
    return pv.read(path)


def save_mesh(mesh: pv.PolyData, path: str):
    pv.save_meshio(path, mesh)


def generate_mesh(
    branches: Iterable[BranchWithRadii],
    mesh_path: str,
    decimate_factor: 0.99,
    gradient_direction: str = "descent",
) -> None:
    voxel_cube = create_empty_voxel_cube_from_branches(branches, [0.6, 0.6, 0.9])
    for _ in range(5):
        voxel_cube.add_padding_layer_all_sides()

    for branch in branches:
        voxel_cube.mark_centerline_in_array(
            branch.coordinates, branch.radii, marking_value=1, radius_padding=0
        )
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)

    mesh = get_surface_mesh(voxel_cube, gradient_direction)
    mesh = mesh.decimate(decimate_factor)
    save_mesh(mesh, mesh_path)


def generate_temp_mesh(
    branches: Iterable[BranchWithRadii], name_base: str, decimate_factor=0.99
) -> str:
    mesh_path = get_temp_mesh_path(name_base)
    generate_mesh(branches, mesh_path, decimate_factor)
    return mesh_path


def get_temp_mesh_path(name_base):
    while True:
        pid = os.getpid()
        nr = int(os.times().elapsed)
        mesh_path = f"{gettempdir()}/{name_base}_{pid}-{nr}.obj"
        if not os.path.exists(mesh_path):
            try:
                open(mesh_path, "x", encoding="utf-8").close()
                break
            except IOError:
                continue
    return mesh_path


def get_high_low_from_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mesh = pv.read(mesh_path)
    bounds = mesh.bounds
    low = np.array([bounds[0], bounds[2], bounds[4]])
    high = np.array([bounds[1], bounds[3], bounds[5]])
    return high, low


def scale_mesh(
    mesh: pv.PolyData, scale_xyz: Tuple[float, float, float], inplace=True
) -> pv.PolyData:
    mesh = mesh.scale(scale_xyz, inplace=inplace)
    return mesh


def rotate_mesh(
    mesh: pv.PolyData,
    rotate_yzx_deg: Tuple[float, float, float],
    inplace=True,
) -> pv.PolyData:
    mesh = mesh.rotate_y(rotate_yzx_deg[0], inplace=inplace)
    mesh = mesh.rotate_z(rotate_yzx_deg[1], inplace=inplace)
    mesh = mesh.rotate_x(rotate_yzx_deg[2], inplace=inplace)
    return mesh
