from setuptools import setup, find_packages
from glob import glob

visu_mesh_data = glob("eve/visualisation/meshes/*")
organic3dv2_data = glob("eve/vesseltree/data/cad/organic3dv2")
organicv2_data = glob("eve/vesseltree/data/cad/organicv2")
setup(
    name="eve",
    version="0.2dev",
    packages=find_packages(),
    data_files=[
        (
            "visu_mesh_data",
            visu_mesh_data,
        ),
        (
            "organicv2_data",
            organicv2_data,
        ),
        (
            "organic3dv2_data",
            organic3dv2_data,
        ),
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pillow",
        "scipy",
        "scikit-image",
        "pyvista",
        "meshio",
        "PyOpenGL",
        "pygame",
        # "pyqt5",
        "matplotlib",
        "pymunk",
        "opencv-python",
    ],
)
