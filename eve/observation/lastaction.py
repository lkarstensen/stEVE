import numpy as np

from .observation import Observation, gym
from ..intervention.intervention import Intervention


class LastAction(Observation):
    def __init__(self, intervention: Intervention, name: str = "last_action") -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.action_space

    def step(self) -> None:
        self.obs = self.intervention.last_action

    def reset(self, episode_nr: int = 0) -> None:
        self.obs = np.zeros_like(self.intervention.last_action, dtype=np.float32)
