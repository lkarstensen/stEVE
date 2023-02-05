import logging

logger = logging.getLogger(__package__)

from .simulation3d import Simulation3D

try:
    from .guidewire import Guidewire
    from .guidewiremp import GuidewireMP
    from .multidevice import MultiDevice
    from .multidevicemp import MutliDeviceMP
    from .ctr import CTR
    from .ctrmp import CTRMP
except ModuleNotFoundError as e:
    logger.warning(f"Sofa probably not installed. Exception: {e}")

from . import device
