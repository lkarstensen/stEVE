from typing import List
import os
import logging
import requests

logger = logging.getLogger("VMR_Download")


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
    logger.info(
        "Downloading vascular models from https://vascularmodel.com/. Please cite appropriately when using for publications."
    )
    for i, model_nr in enumerate(model_nrs):
        path_model = os.path.join(path_vmr, model_nr)
        if not os.path.exists(path_model):
            os.mkdir(path_model)

        info_log = f"Downloading {model_nr} to {path_model}/ ({i+1}/{n_models})"
        logger.info(info_log)

        license_url = f"https://vascularmodel.com/svprojects/{model_nr}/LICENSE"
        pdf_Url = f"https://vascularmodel.com/svprojects/{model_nr}/{model_nr}.pdf"
        path_LICENSE = os.path.join(path_model, "LICENSE")
        if not os.path.isfile(path_LICENSE):
            logger.info("Downloading License")
            _download(license_url, path_LICENSE)
        path_pdf = os.path.join(path_model, f"{model_nr}.pdf")
        if not os.path.isfile(path_pdf):
            logger.info("Downloading overview pdf")
            _download(pdf_Url, path_pdf)

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
            logger.info("Downloading vtp mesh")
            _download(vtp_url, vtp_path)
        if not os.path.isfile(vtu_path):
            logger.info("Downloading vtu mesh")
            _download(vtu_url, vtu_path)

        paths_url = f"https://vascularmodel.com/svprojects/{model_nr}/Paths/"
        data = requests.get(paths_url, timeout=10)
        text = data.text

        arterie_names = []
        while ".pth" in text:
            first_pth = text.find(".pth")
            start = text.find('">', first_pth) + 2
            end = text.find("</a>", first_pth)
            arterie_names.append(text[start:end])
            text = text[end:]
        path_paths = os.path.join(path_model, "Paths")
        if not os.path.exists(path_paths):
            os.mkdir(path_paths)

        for arterie in arterie_names:
            arterie_path = os.path.join(path_paths, arterie)
            if not os.path.isfile(arterie_path):
                info_log = f"Downloading arterie {arterie}"
                logger.info(info_log)
                arterie_url = paths_url + arterie
                _download(
                    arterie_url,
                    arterie_path,
                )

    info_log = f"Download of {n_models} finished."
    logger.info(info_log)


def _download(vtu_url, vtu_path):
    data = requests.get(vtu_url, timeout=60)
    with open(vtu_path, "wb") as file:
        file.write(data.content)
    del data
