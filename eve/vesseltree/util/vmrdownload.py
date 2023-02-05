from typing import List
import urllib.request
import os


def download_vmr_files(model_nrs: List[str]):

    n_models = len(model_nrs)
    path_dir = os.path.dirname(os.path.realpath(__file__))
    path_vessel_tree, _ = os.path.split(path_dir)
    path_data = os.path.join(path_vessel_tree, "data")
    path_vmr = os.path.join(path_data, "vmr")
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    if not os.path.exists(path_vmr):
        os.mkdir(path_vmr)
    print(
        "Downloading vascular models from https://vascularmodel.com/. Please cite appropriately when using for publications."
    )
    for i, model_nr in enumerate(model_nrs):
        path_model = os.path.join(path_vmr, model_nr)
        if not os.path.exists(path_model):
            os.mkdir(path_model)

        print(f"Downloading {i+1}/{n_models}: {model_nr} to {path_model}/")

        license_url = f"https://vascularmodel.com/svprojects/{model_nr}/LICENSE"
        pdf_Url = f"https://vascularmodel.com/svprojects/{model_nr}/{model_nr}.pdf"
        path_LICENSE = os.path.join(path_model, "LICENSE")
        if not os.path.isfile(path_LICENSE):
            urllib.request.urlretrieve(
                license_url,
                path_LICENSE,
            )
        path_pdf = os.path.join(path_model, f"{model_nr}.pdf")
        if not os.path.isfile(path_pdf):
            urllib.request.urlretrieve(
                pdf_Url,
                path_pdf,
            )

        path_model_mesh = os.path.join(path_model, "Meshes")
        if not os.path.exists(path_model_mesh):
            os.mkdir(path_model_mesh)
        base_mesh_url = (
            f"https://vascularmodel.com/svprojects/{model_nr}/Meshes/{model_nr}"
        )
        vtp_url = base_mesh_url + ".vtp"
        vtu_url = base_mesh_url + ".vtu"
        vtp_path = os.path.join(path_model_mesh, model_nr + ".vtp")
        vtu_path = os.path.join(path_model_mesh, model_nr + ".vtu")
        if not os.path.isfile(vtp_path):
            urllib.request.urlretrieve(
                vtp_url,
                vtp_path,
            )
        if not os.path.isfile(vtu_path):
            urllib.request.urlretrieve(
                vtu_url,
                vtu_path,
            )

        print(f"Download {i+1}/{n_models}: {model_nr} finished.")
