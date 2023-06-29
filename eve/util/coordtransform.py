from typing import Tuple
import numpy as np


def _get_rot_matrix(image_rot_zx):
    rot_z = -image_rot_zx[0] * np.pi / 180
    rot_x = -image_rot_zx[1] * np.pi / 180
    rotation_matrix_z = np.array(
        [
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1],
        ],
    )

    rotation_matrix_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)],
        ],
    )
    rotation_matrix = np.matmul(rotation_matrix_z, rotation_matrix_x)
    return rotation_matrix


def vessel_cs_to_tracking3d(
    array: np.ndarray,
    image_rot_zx: Tuple[float, float],
    image_center: Tuple[float, float],
    field_of_view: Tuple[float, float],
):
    rot_matrix = _get_rot_matrix(image_rot_zx)
    new_array = np.matmul(rot_matrix, array.T).T

    image_center = np.array(image_center)
    image_center_rot_cs = np.matmul(rot_matrix, image_center.T).T
    new_array = new_array - image_center_rot_cs
    if field_of_view is not None:
        fov = field_of_view
        low = np.array([-fov[0] / 2, -np.inf, -fov[1] / 2])
        high = -low
        low_bound = np.any(new_array < low, axis=-1)
        high_bound = np.any(new_array > high, axis=-1)
        out_of_bounds = low_bound + high_bound
        new_array = new_array[~out_of_bounds]
    return new_array.astype(dtype=np.float32)


def tracking3d_to_vessel_cs(
    array: np.ndarray,
    image_rot_zx: Tuple[float, float],
    image_center: Tuple[float, float],
):
    rot_matrix = _get_rot_matrix(image_rot_zx)

    image_center = np.array(image_center)
    image_center_rot_cs = np.matmul(rot_matrix, image_center.T).T
    new_array = array + image_center_rot_cs
    # rot_matrix.T because this is the inverse to vessel_cs_to_tracking3d (what the rot matrix was made for)
    new_array = np.matmul(rot_matrix.T, new_array.T).T
    return new_array.astype(dtype=np.float32)


def tracking3d_to_2d(tracking3d: np.ndarray):
    return np.delete(tracking3d, 1, -1)
