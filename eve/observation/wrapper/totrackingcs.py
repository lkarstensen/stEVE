from typing import Optional

from ...intervention import Intervention
from ..observation import Observation, gym


class ToTrackingCS(Observation):
    def __init__(
        self,
        wrapped_obs: Observation,
        intervention: Intervention,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        self.intervention = intervention
        self.wrapped_obs = wrapped_obs

    @property
    def space(self) -> gym.spaces.Box:
        high = self.wrapped_obs.space.high
        high = self.intervention.vessel_cs_to_tracking_cs(high)
        low = self.wrapped_obs.space.low
        low = self.intervention.vessel_cs_to_tracking_cs(low)
        return gym.spaces.Box(low=low, high=high)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._get_obs()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._get_obs()

    def _get_obs(self):
        obs = self.wrapped_obs.obs
        obs = self.intervention.vessel_cs_to_tracking_cs(obs)
        self.obs = obs
