from typing import Optional
import numpy as np

from ..observation import Observation, gym
from ...vesseltree import VesselTree


class RelativeToInsertionPoint(Observation):
    def __init__(
        self,
        vessel_tree: VesselTree,
        wrapped_obs: Observation,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_obs.name
        super().__init__(name)
        if wrapped_obs.obs.shape[-1] != 3:
            raise ValueError(
                f"{self.__class__} can only be used with 3 dimensional States.\
                    Not with {wrapped_obs.obs.shape[-1]} Dimensions"
            )
        self.vessel_tree = vessel_tree

        self.wrapped_obs = wrapped_obs
        self._insertion_point = None

    @property
    def space(self) -> gym.spaces.Box:
        wrapped_high = self.wrapped_obs.space.high
        vessel_low = self.vessel_tree.bounding_box.low
        vessel_low = np.full_like(wrapped_high, vessel_low, dtype=np.float32)
        high = wrapped_high - vessel_low
        wrapped_low = self.wrapped_obs.space.low
        vessel_high = self.vessel_tree.bounding_box.high
        vessel_high = np.full_like(wrapped_low, vessel_high, dtype=np.float32)
        low = wrapped_low - vessel_high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self._calc_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._insertion_point = self.vessel_tree.insertion.position
        self._calc_state()

    def _calc_state(self):
        state = self.wrapped_obs()
        subtrahend = np.full_like(state, self._insertion_point, dtype=np.float32)
        self.obs = state - subtrahend
