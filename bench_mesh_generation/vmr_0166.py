from time import perf_counter

import os
import numpy as np
import pyvista as pv

from pykdtree.kdtree import KDTree  # pylint: disable=no-name-in-module
import eve
from eve.vesseltree.util.meshing import get_surface_mesh, save_mesh
from eve.vesseltree.util.voxelcube import (
    create_voxel_cube_from_mesh,
    VoxelCube,
)
from eve.vesseltree.util.branch import BranchWithRadii


def extend_branch_end(branch: BranchWithRadii, start_end: str, length: int):
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

    # if start_end == "start":
    #     new_coordinates = np.vstack([new_points, branch.coordinates])
    #     new_radii = np.concatenate([new_radii, branch.radii], axis=0)
    # else:
    #     new_coordinates = np.vstack([branch.coordinates, new_points])
    #     new_radii = np.concatenate([branch.radii, new_radii], axis=0)

    return BranchWithRadii(branch.name, new_points, new_radii)


def make_printable_vmr(
    rot_z: float,
    rot_x: float,
    vmr_folder: str,
    z_split: float,
    z_remove_lower: float = None,
    z_remove_upper: float = None,
):
    arch = eve.vesseltree.VMR(
        vmr_folder,
        -1,
        -2,
        rotate_yzx_deg=[0, rot_z, rot_x],
        insertion_vessel_name="lva",
    )
    arch.reset()

    start = perf_counter()

    _, patient = os.path.split(vmr_folder)

    file = os.path.join(vmr_folder, "Meshes", patient) + ".vtu"

    mesh = pv.read(file)
    mesh.scale([10, 10, 10], inplace=True)
    mesh.rotate_z(rot_z, inplace=True)
    mesh.rotate_x(rot_x, inplace=True)
    cube = create_voxel_cube_from_mesh(mesh, [0.2, 0.2, 0.2])
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

    end_extensions = []

    for branch in arch:
        start = branch.coordinates[0]

        if arch.at_tree_end(start):
            new_branch = extend_branch_end(branch, "start", 6)
            end_extensions.append(new_branch)

        end = branch.coordinates[-1]

        if arch.at_tree_end(end):
            new_branch = extend_branch_end(branch, "end", 6)
            end_extensions.append(new_branch)

    for branch in end_extensions:
        radius = 5 if branch.name == "aorta" else 1.5
        wall_model.mark_centerline_in_array(
            branch.coordinates, marking_value=0, cl_radii=radius
        )

    wall_model.gaussian_smooth(1)
    wall_model.gaussian_smooth(1)
    wall_model.gaussian_smooth(1)

    mesh = get_surface_mesh(wall_model, "ascent")
    mesh = mesh.decimate(0.9, inplace=True)
    cwd = os.getcwd()
    save_mesh(
        mesh,
        os.path.join(cwd, f"{patient}_printmesh_full_{rot_z=}_{rot_x=}.obj"),
    )
    if z_split is not None:
        z_split_idx = int(z_split / wall_model.spacing[2])

        lower_model = VoxelCube(
            wall_model.value_array.copy(),
            wall_model.spacing,
            wall_model.world_offset,
        )
        lower_model.value_array[:, :, z_split_idx:] = 0
        if z_remove_lower is not None:
            z_remove_lower_idx = int(z_remove_lower / wall_model.spacing[2])
            lower_model.value_array[:, :, :z_remove_lower_idx] = 0

        upper_model = VoxelCube(
            wall_model.value_array.copy(),
            wall_model.spacing,
            wall_model.world_offset,
        )
        upper_model.value_array[:, :, :z_split_idx] = 0
        if z_remove_upper is not None:
            z_remove_upper_idx = int(z_remove_upper / wall_model.spacing[2])
            lower_model.value_array[:, :, z_remove_upper_idx:] = 0

        mesh = get_surface_mesh(lower_model, "ascent")
        mesh.decimate(0.9, inplace=True)
        save_mesh(
            mesh,
            os.path.join(cwd, f"{patient}_printmesh_lower_{rot_z=}_{rot_x=}.obj"),
        )

        mesh = get_surface_mesh(upper_model, "ascent")
        mesh.decimate(0.9, inplace=True)
        save_mesh(
            mesh,
            os.path.join(cwd, f"{patient}_printmesh_upper_{rot_z=}_{rot_x=}.obj"),
        )


if __name__ == "__main__":
    make_printable_vmr(
        180,
        0,
        "/Users/lennartkarstensen/stacie/eve_toolbox/eve/vesseltree/data/vmr/0166_0001",
        z_split=None,
        z_remove_lower=None,
        z_remove_upper=None,
    )
