from .reward import Reward
from ..intervention.intervention import Intervention


class InsertionLengthRelativeDelta(Reward):
    def __init__(
        self,
        intervention: Intervention,
        device_id: int,
        relative_to_device_id: int,
        factor: float,
        lower_clearance: float,
        upper_clearance: float,
    ) -> None:
        self.intervention = intervention
        self.device_id = device_id
        self.relative_to_device_id = relative_to_device_id
        self.factor = factor
        self.lower_clearance = lower_clearance
        self.upper_clearance = upper_clearance
        self._last_relative_length = 0.0

    def step(self) -> None:
        inserted_lengths = self.intervention.device_lengths_inserted
        relative_length = (
            inserted_lengths[self.device_id]
            - inserted_lengths[self.relative_to_device_id]
        )

        if self.upper_clearance > relative_length > self.lower_clearance:
            self.reward = 0.0
        else:
            delta = abs(relative_length) - abs(self._last_relative_length)
            self.reward = delta * self.factor
        self._last_relative_length = relative_length

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
