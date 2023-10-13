from typing import List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym

from .vesseltree import VesselTree, Insertion
from .util.branch import (
    calc_branching,
    Branch,
    BranchWithRadii,
    calc_branching_with_radii,
    rotate_array,
)

from .util.branch import rotate_branches as rot_branches_func
from .util.meshing import (
    rotate_mesh,
    scale_mesh,
    get_temp_mesh_path,
    load_mesh,
    save_mesh,
    get_high_low_from_mesh,
)


class FromMesh(VesselTree):
    def __init__(
        self,
        mesh: str,
        insertion_position: Tuple[float, float, float],
        insertion_direction: Tuple[float, float, float],
        branch_list: Optional[List[Union[Branch, BranchWithRadii]]] = None,
        approx_branch_radii: Optional[Union[List[float], float]] = None,
        visu_mesh: Optional[str] = None,
        scaling_xyz: Optional[Tuple[float, float, float]] = None,
        rotation_yzx_deg: Optional[Tuple[float, float, float]] = None,
        rotate_branches: bool = True,
        rotate_ip: bool = True,
    ) -> None:
        self.mesh = mesh
        self.visu_mesh = visu_mesh
        self.insertion_position = insertion_position
        self.insertion_direction = insertion_direction
        self.branch_list = branch_list

        self.scaling_xyz = scaling_xyz or [1.0, 1.0, 1.0]
        self.rotation_yzx_deg = rotation_yzx_deg or [0.0, 0.0, 0.0]
        self.rotate_branches = rotate_branches

        temp_mesh_path = get_temp_mesh_path("mesh_from_file")
        mesh = load_mesh(mesh)
        mesh = rotate_mesh(mesh, self.rotation_yzx_deg)
        mesh = scale_mesh(mesh, self.scaling_xyz)
        save_mesh(mesh, temp_mesh_path)
        self.mesh_path = temp_mesh_path

        if visu_mesh is not None:
            temp_mesh_path = get_temp_mesh_path("mesh_from_file")
            mesh = load_mesh(visu_mesh)
            mesh = rotate_mesh(mesh, self.rotation_yzx_deg)
            mesh = scale_mesh(mesh, self.scaling_xyz)
            save_mesh(mesh, temp_mesh_path)
        self.visu_mesh_path = temp_mesh_path
        if rotate_ip:
            self.insertion = Insertion(
                rotate_array(
                    np.array(insertion_position),
                    rotation_yzx_deg[0],
                    rotation_yzx_deg[1],
                    rotation_yzx_deg[2],
                ),
                rotate_array(
                    np.array(insertion_direction),
                    rotation_yzx_deg[0],
                    rotation_yzx_deg[1],
                    rotation_yzx_deg[2],
                ),
            )
        else:
            self.insertion = Insertion(insertion_position, insertion_direction)

        if branch_list is not None:
            if rotate_branches:
                branch_list = rot_branches_func(
                    branches=branch_list, rotate_yzx_deg=rotation_yzx_deg
                )
            branch_highs = [branch.high for branch in branch_list]
            high = np.max(branch_highs, axis=0)
            branch_lows = [branch.low for branch in branch_list]
            low = np.min(branch_lows, axis=0)
            if isinstance(branch_list[0], BranchWithRadii):
                self.branching_points = calc_branching_with_radii(branch_list)
            else:
                self.branching_points = calc_branching(branch_list, approx_branch_radii)
            centerline_coordinates = [branch.coordinates for branch in branch_list]
            self.centerline_coordinates = np.concatenate(centerline_coordinates)
        else:
            high, low = get_high_low_from_mesh(temp_mesh_path)
            self.branching_points = None
            self.centerline_coordinates = np.zeros((1, 3), dtype=np.float32)

        self.branches = branch_list
        self.approx_branch_radii = approx_branch_radii
        self.coordinate_space = gym.spaces.Box(low, high)
        self.coordinate_space_episode = self.coordinate_space

    def step(self) -> None:
        ...

    def reset(self, episode_nr=0, seed: int = None) -> None:
        ...
