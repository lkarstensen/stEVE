from eve.vesseltree.util.vmrdownload import download_vmr_files
import logging

logging.basicConfig(level=logging.INFO)

# Add all VMR Models that are useful for this project:
models = [
    "0094_0001",
    "0095_0001",
    "0105_0001",
    "0111_0001",
    "0131_0000",
    "0154_0001",
    "0175_0000",
    "0176_0000",
]

download_vmr_files(models)
