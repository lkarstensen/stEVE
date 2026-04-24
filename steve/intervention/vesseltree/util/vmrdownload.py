import os
import logging
import requests

logger = logging.getLogger("VMR_Download")


def download_vmr_files(model: str) -> str:
    path_util = os.path.dirname(os.path.realpath(__file__))
    path_vessel_tree, _ = os.path.split(path_util)
    path_intervention, _ = os.path.split(path_vessel_tree)
    path_eve, _ = os.path.split(path_intervention)
    path_lib_base, _ = os.path.split(path_eve)
    path_data = os.path.join(path_lib_base, ".data")
    path_vmr = os.path.join(path_data, "vmr")
    path_model = os.path.join(path_vmr, model)

    if os.path.exists(path_model):
        log_text = f"{model} folder found at {path_model}. Not Downloading again."
        logger.info(log_text)
        return path_model

    if not os.path.exists(path_data):
        os.mkdir(path_data)
    if not os.path.exists(path_vmr):
        os.mkdir(path_vmr)
    if not os.path.exists(path_model):
        os.mkdir(path_model)

    logger.info(
        "Downloading vascular models from https://vascularmodel.com/. Please cite appropriately when using for publications."
    )

    info_log = f"Downloading {model} to {path_model}"
    logger.info(info_log)

    license_url = f"https://vascularmodel.com/svprojects/{model}/LICENSE"
    pdf_Url = f"https://vascularmodel.com/svprojects/{model}/{model}.pdf"
    path_LICENSE = os.path.join(path_model, "LICENSE")
    if not os.path.isfile(path_LICENSE):
        logger.info("Downloading License")
        _download(license_url, path_LICENSE)
    path_pdf = os.path.join(path_model, f"{model}.pdf")
    if not os.path.isfile(path_pdf):
        logger.info("Downloading overview pdf")
        _download(pdf_Url, path_pdf)

    path_model_mesh = os.path.join(path_model, "Meshes")
    if not os.path.exists(path_model_mesh):
        os.mkdir(path_model_mesh)

    meshes_url = f"https://vascularmodel.com/svprojects/{model}/Meshes/"

    data = requests.get(meshes_url, timeout=10)
    text = data.text

    vtp_mesh = text.find(".vtp")
    start = text.find('">', vtp_mesh) + 2
    end = text.find("</a>", vtp_mesh)
    vtp_mesh = text[start:end]
    vtp_url = meshes_url + vtp_mesh

    vtu_mesh = text.find(".vtu")
    start = text.find('">', vtu_mesh) + 2
    end = text.find("</a>", vtu_mesh)
    vtu_mesh = text[start:end]
    vtu_url = meshes_url + vtu_mesh

    vtp_path = os.path.join(path_model_mesh, model + ".vtp")
    vtu_path = os.path.join(path_model_mesh, model + ".vtu")
    if not os.path.isfile(vtp_path):
        logger.info("Downloading vtp mesh")
        _download(vtp_url, vtp_path)
    if not os.path.isfile(vtu_path):
        logger.info("Downloading vtu mesh")
        _download(vtu_url, vtu_path)

    paths_url = f"https://vascularmodel.com/svprojects/{model}/Paths/"
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

    for artery in arterie_names:
        artery_path = os.path.join(path_paths, artery)
        if not os.path.isfile(artery_path):
            info_log = f"Downloading {artery}"
            logger.info(info_log)
            arterie_url = paths_url + artery
            _download(
                arterie_url,
                artery_path,
            )

    info_log = f"Download of {model} finished."
    logger.info(info_log)
    return path_model


def _download(vtu_url, vtu_path):
    data = requests.get(vtu_url, timeout=60)
    with open(vtu_path, "wb") as file:
        file.write(data.content)
    del data
