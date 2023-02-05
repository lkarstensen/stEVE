from time import perf_counter
from pykdtree.kdtree import KDTree
import os
import numpy as np
import pyvista as pv

import eve
from .meshing import get_surface_mesh, save_mesh
from .voxelcube import (
    create_voxel_cube_from_mesh,
    VoxelCube,
    create_empty_voxel_cube_from_branches,
)
from .branch import Branch
from ..vesseltree import VesselTree


def print_obj_from_selfmade(vesseltree: VesselTree, mesh_path: str):

    voxel_cube = create_empty_voxel_cube_from_branches(vesseltree, [0.3, 0.3, 0.3])
    for _ in range(100):
        voxel_cube.add_padding_layer_all_sides()

    ip_tube, ip_lumen = get_branch_start_tube_and_lumen(
        vesseltree.centerline_tree.branches[0], 5, 3, wall_thickness=2
    )

    endings_tubes = []
    endings_lumen = []

    for branch in vesseltree:
        if branch.name in [
            "left common carotid artery",
            "right common carotid artery",
            "left subclavian artery",
            "right subclavian artery",
        ]:
            tube, lumen = get_branch_end_tube_and_lumen(branch, 3, 2, 1.5)
            endings_tubes.append(tube)
            endings_lumen.append(lumen)
        if branch.name == "aorta":
            tube, lumen = get_branch_end_tube_and_lumen(branch, 5, 3, 1.5)
            endings_tubes.append(tube)
            endings_lumen.append(lumen)

    for branch in vesseltree.centerline_tree.branches:
        voxel_cube.mark_centerline_in_array(branch, marking_value=1, radius_padding=3)

    for branch in vesseltree.centerline_tree.branches:
        voxel_cube.mark_centerline_in_array(branch, marking_value=0, radius_padding=0)

    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)

    voxel_cube.mark_centerline_in_array(ip_tube, marking_value=1, radius_padding=0)
    for tube in endings_tubes:
        voxel_cube.mark_centerline_in_array(tube, marking_value=1, radius_padding=0)
    voxel_cube.mark_centerline_in_array(ip_lumen, marking_value=0, radius_padding=0)
    for lumen in endings_lumen:
        voxel_cube.mark_centerline_in_array(lumen, marking_value=0, radius_padding=0)

    # voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1.0)
    voxel_cube.gaussian_smooth(0.7)

    mesh = get_surface_mesh(voxel_cube)
    # mesh = mesh.decimate_pro(0.9)
    mesh = mesh.decimate(0.95)
    save_mesh(mesh, mesh_path)


def get_branch_start_tube_and_lumen(
    branch: Branch, length: float, diameter: float, wall_thickness: float
):
    first_point = branch.coordinates[0]
    second_point = branch.coordinates[1]
    direction = first_point - second_point
    direction = direction / np.linalg.norm(direction)

    return get_tube_and_lumen(first_point, direction, length, diameter, wall_thickness)


def get_branch_end_tube_and_lumen(
    branch: Branch, length: float, diameter: float, wall_thickness: float
):
    last_point = branch.coordinates[-1]
    second_to_last_point = branch.coordinates[-2]
    direction = last_point - second_to_last_point
    direction = direction / np.linalg.norm(direction)

    return get_tube_and_lumen(last_point, direction, length, diameter, wall_thickness)


def get_tube_and_lumen(
    point: np.ndarray,
    direction: np.ndarray,
    length: float,
    diameter: float,
    wall_thickness: float,
):
    tube_start = point + direction * (point.radius + diameter / 2 + wall_thickness)
    tube_end = tube_start + direction * length
    tube_length = (tube_start - tube_end).length

    lumen_start = point
    lumen_end = tube_end + direction * (diameter / 2 + wall_thickness)
    lumen_length = (lumen_start - lumen_end).length

    n_tube_points = 10 * int(np.ceil(tube_length))
    n_lumen_points = 10 * int(np.ceil(lumen_length))

    x_tube = np.linspace(tube_start.x, tube_end.x, num=n_tube_points)
    y_tube = np.linspace(tube_start.y, tube_end.y, num=n_tube_points)
    z_tube = np.linspace(tube_start.z, tube_end.z, num=n_tube_points)

    x_lumen = np.linspace(lumen_start.x, lumen_end.x, num=n_lumen_points)
    y_lumen = np.linspace(lumen_start.y, lumen_end.y, num=n_lumen_points)
    z_lumen = np.linspace(lumen_start.z, lumen_end.z, num=n_lumen_points)

    tube_points = []
    for i in range(n_tube_points):
        point = CenterlinePoint(
            x_tube[i], y_tube[i], z_tube[i], diameter / 2 + wall_thickness
        )
        tube_points.append(point)

    lumen_points = []
    for i in range(n_lumen_points):
        point = CenterlinePoint(x_lumen[i], y_lumen[i], z_lumen[i], diameter / 2)
        lumen_points.append(point)

    return CenterlineBranch(tube_points), CenterlineBranch(lumen_points)


def extend_branch_end(branch: Branch, start_end: str, length: int):

    if start_end == "start":
        coord_idx = 0
        direction_idx = 1
    else:
        coord_idx = -1
        direction_idx = -2
    coord_start = branch.coordinates[coord_idx]
    direction = branch.coordinates[coord_idx] - branch.coordinates[direction_idx]
    direction = direction / np.linalg.norm(direction)
    coord_end = coord_start + length * direction

    radius = branch.radii[coord_idx]

    new_points = np.linspace(
        coord_end, coord_start, num=int(np.ceil(length)), endpoint=False
    )
    if not start_end == "start":
        new_points = np.flip(new_points, axis=0)

    n_points = new_points.shape[0]

    new_radii = np.ones([n_points]) * radius

    if start_end == "start":
        new_coordinates = np.vstack([new_points, branch.coordinates])
        new_radii = np.concatenate([new_radii, branch.radii], axis=0)
    else:
        new_coordinates = np.vstack([branch.coordinates, new_points])
        new_radii = np.concatenate([branch.radii, new_radii], axis=0)

    return Branch(branch.name, new_coordinates, new_radii)


def make_printable_vmr(rot_z: float, rot_x: float, vmr_folder: str, z_split: float):

    arch = eve.vesseltree.VMR(
        vmr_folder,
        -1,
        -2,
        rotate_yzx_deg=[0, rot_z, rot_x],
    )
    arch.reset()

    start = perf_counter()

    _, patient = os.path.split(vmr_folder)

    file = os.path.join(vmr_folder, "Meshes", patient) + ".vtu"

    mesh = pv.read(file)
    mesh.scale([10, 10, 10], inplace=True)
    mesh.rotate_z(rot_z, inplace=True)
    mesh.rotate_x(rot_x, inplace=True)
    cube = create_voxel_cube_from_mesh(mesh, [0.3, 0.3, 0.3])
    cube.add_padding_layer_all_sides(n_layers=12)

    coords_flat = cube.voxel_coords.reshape(-1, 3)

    tree = KDTree(mesh.points.astype(np.double))

    dist_to_mesh, _ = tree.query(coords_flat)

    in_boundary = dist_to_mesh < 1.5

    in_boundary = in_boundary.astype(np.float32)

    in_boundary = in_boundary.reshape(cube.value_array.shape)

    wall = in_boundary - cube.value_array

    wall_model = VoxelCube(wall, cube.spacing, cube.world_offset)

    print(f"time: {perf_counter()-start}")

    new_branches = []

    for branch in arch:

        new_branch = extend_branch_end(branch, "end", 6)
        if branch.name == "aorta":
            new_branch = extend_branch_end(new_branch, "start", 6)
        new_branches.append(new_branch)

    for branch in new_branches:
        wall_model.mark_centerline_in_array(branch, marking_value=0, custom_radius=1.6)

    wall_model.gaussian_smooth(1)
    wall_model.gaussian_smooth(1)
    wall_model.gaussian_smooth(1)

    mesh = get_surface_mesh(wall_model, "ascent")
    mesh = mesh.decimate(0.9, inplace=True)
    save_mesh(
        mesh,
        f"/Users/lennartkarstensen/stacie/eve/{patient}_printmesh_full_{rot_z=}_{rot_x=}.obj",
    )

    z_split_idx = int(z_split / wall_model.spacing[2])

    lower_model = VoxelCube(
        wall_model.value_array.copy(),
        wall_model.spacing,
        wall_model.world_offset,
    )
    lower_model.value_array[:, :, z_split_idx:] = 0

    upper_model = VoxelCube(
        wall_model.value_array.copy(),
        wall_model.spacing,
        wall_model.world_offset,
    )
    upper_model.value_array[:, :, :z_split_idx] = 0

    mesh = get_surface_mesh(lower_model, "ascent")
    mesh.decimate(0.9, inplace=True)
    save_mesh(
        mesh,
        f"/Users/lennartkarstensen/stacie/eve/{patient}_printmesh_lower_{rot_z=}_{rot_x=}.obj",
    )

    mesh = get_surface_mesh(upper_model, "ascent")
    mesh.decimate(0.9, inplace=True)
    save_mesh(
        mesh,
        f"/Users/lennartkarstensen/stacie/eve/{patient}_printmesh_upper_{rot_z=}_{rot_x=}.obj",
    )
