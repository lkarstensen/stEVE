from .start import Start
from ..intervention import Intervention


class MaxDeviceLength(Start):
    def __init__(self, intervention: Intervention, max_length: float):
        self.intervention = intervention
        self.max_length = max_length

    def reset(self, episode_nr: int = 0) -> None:
        reset = False
        for inserted_length in self.intervention.device_lengths_inserted.values():
            if inserted_length > self.max_length:
                reset = True
        if reset:
            self.intervention.reset_devices()
