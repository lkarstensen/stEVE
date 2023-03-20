import logging

logger = logging.getLogger(__package__)
from .visualisation import Visualisation

try:
    from .plt3d import PLT3D
    from .sofapygame import SofaPygame
    from .fromimaging import FromImaging
    from .fromstate import FromState
except ImportError as e:
    logger.warning(
        f"some visualisation not working. Probably headless or missing dependency. Here's the exception: {e}"
    )

from .dummy import Dummy as VisualisationDummy
