import numpy as np

from ..intervention import Intervention
from .observation import Observation, gym


class InsertionLengthRelative(Observation):
    def __init__(
        self,
        intervention: Intervention,
        instrument_idx: int,
        relative_to_instrument_idx: int,
        name: str = None,
    ) -> None:
        name = (
            name
            or f"Instrument_length_{instrument_idx}_relative_to_{relative_to_instrument_idx}"
        )
        super().__init__(name)
        self.intervention = intervention
        self.instrument_idx = instrument_idx
        self.relative_to_instrument_idx = relative_to_instrument_idx

    @property
    def space(self) -> gym.spaces.Box:
        high = self.intervention.instrument_lengths_maximum[self.instrument_idx]
        high = np.array(high, dtype=np.float32)
        low = -self.intervention.instrument_lengths_maximum[
            self.relative_to_instrument_idx
        ]
        low = np.array(low, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        inserted_lengths = self.intervention.instrument_lengths_inserted
        relative_length = (
            inserted_lengths[self.instrument_idx]
            - inserted_lengths[self.relative_to_instrument_idx]
        )
        self.obs = np.array(relative_length, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
