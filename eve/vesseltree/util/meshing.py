from tempfile import gettempdir
import os
from typing import Iterable
import numpy as np
import pyvista as pv

from .voxelcube import VoxelCube, create_empty_voxel_cube_from_branches, Branch


def get_surface_mesh(
    voxel_cube: VoxelCube, gradient_direction: str = "descent"
) -> pv.PolyData:
    from skimage import measure

    vertices, faces, _, _ = measure.marching_cubes(
        voxel_cube.value_array,
        None,
        spacing=voxel_cube.spacing,
        gradient_direction=gradient_direction,
    )
    faces_nr_points = np.ones((faces.shape[0], 1), dtype=np.int64)
    faces_nr_points *= 3
    faces = np.concatenate((faces_nr_points, faces), axis=1)
    mesh = pv.PolyData(vertices, faces)
    mesh.translate(voxel_cube.world_offset, inplace=True)
    return mesh


def save_mesh(mesh: pv.PolyData, path: str):
    pv.save_meshio(path, mesh)


def generate_mesh(
    branches: Iterable[Branch],
    mesh_path: str,
    decimate_factor: 0.99,
    gradient_direction: str = "descent",
) -> None:
    voxel_cube = create_empty_voxel_cube_from_branches(branches, [0.6, 0.6, 0.9])
    for _ in range(5):
        voxel_cube.add_padding_layer_all_sides()

    for branch in branches:
        voxel_cube.mark_centerline_in_array(branch, marking_value=1, radius_padding=0)
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)

    mesh = get_surface_mesh(voxel_cube, gradient_direction)
    mesh = mesh.decimate(decimate_factor)
    save_mesh(mesh, mesh_path)


def generate_temp_mesh(
    branches: Iterable[Branch], name_base: str, decimate_factor=0.99
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
