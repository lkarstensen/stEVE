from .reward import Reward
from ..intervention.intervention import Intervention


class LastAction(Reward):
    def __init__(
        self, intervention: Intervention, action_idx: int, factor: float
    ) -> None:
        self.intervention = intervention

        self.action_idx = action_idx
        self.factor = factor

        if self.action_idx > len(self.intervention.last_action) - 1:
            raise ValueError(
                f"Speed Index needs to map to the speed Tuple of the device. speed_idx: {self.action_idx} cannot be used. It needs to be between 0 and {len(self.intervention.speed)-1} speed values available"
            )

    def step(self) -> None:
        last_action = self.intervention.last_action
        self.reward = last_action[self.action_idx] * self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
