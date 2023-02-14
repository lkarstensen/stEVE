import pyvista as pv


def convert_vtp_to_obj(
    scaling_facto: int, vtp_path: str, obj_path: str, rotate_yzx_deg
):
    mesh = pv.read(vtp_path)
    # mesh.flip_normals()
    mesh.scale([scaling_facto, scaling_facto, scaling_facto], inplace=True)
    if rotate_yzx_deg is not None:
        mesh.rotate_y(rotate_yzx_deg[0], inplace=True)
        mesh.rotate_z(rotate_yzx_deg[1], inplace=True)
        mesh.rotate_x(rotate_yzx_deg[2], inplace=True)
    # mesh.decimate(0.9, inplace=True)
    pv.save_meshio(obj_path, mesh)
