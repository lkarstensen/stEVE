from typing import List
import numpy as np

from . import Start
from ..intervention.intervention import Intervention


class RandomAdvance(Start):
    def __init__(
        self,
        intervention: Intervention,
        t_advance: float,
        action_low: List[float],
        action_high: List[float],
    ) -> None:
        self.intervention = intervention
        self.t_advance = t_advance
        self.action_low = np.array(action_low)
        self.actin_high = np.array(action_high)

    def reset(self, episode_nr: int = 0) -> None:
        self.intervention.reset_devices()
        n_steps = int(self.t_advance * self.intervention.image_frequency)

        for _ in range(n_steps):
            action = np.random.uniform(self.action_low, self.actin_high)
            self.intervention.step(action)
        for _ in range(3):
            self.intervention.step(np.zeros_like(action))
