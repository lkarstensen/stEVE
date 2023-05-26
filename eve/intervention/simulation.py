from typing import Optional, Tuple
from .intervention import Intervention
from .sofacore import SOFACore


class Simulation(Intervention):
    sofa_core: SOFACore
    image_frequency: float
    image_rot_zx: Tuple[float, float]
    image_center: Optional[Tuple[float, float, float]]
    field_of_view: Optional[Tuple[float, float]]
