import numpy as np
from .start import Start
from ..intervention.intervention import Intervention


class MaxDeviceLength(Start):
    def __init__(self, intervention: Intervention, max_length: float):
        self.intervention = intervention
        self.max_length = max_length

    def reset(self, episode_nr: int = 0) -> None:
        if np.any(self.intervention.device_lengths_inserted > self.max_length):
            self.intervention.reset_devices()
