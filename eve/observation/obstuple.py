from abc import ABC
from typing import List, Tuple

from .observation import Observation, gym, np


class ObsTuple(ABC):
    def __init__(
        self, observations: List[Observation], name: str = "observation_tuple"
    ) -> None:
        self.name = name
        self.obs: Tuple[np.ndarray] = ()
        self.observations = observations

    @property
    def space(self) -> gym.spaces.Tuple:
        spaces = [obs.space for obs in self.observations]
        return gym.spaces.Tuple(spaces)

    def step(self) -> None:
        new_obs = []
        for wrapped_obs in self.observations:
            wrapped_obs.step()
            new_obs.append(wrapped_obs.obs)
        self.obs = tuple(new_obs)

    def reset(self, episode_nr: int = 0) -> None:
        new_obs = []
        for wrapped_obs in self.observations:
            wrapped_obs.reset(episode_nr)
            new_obs.append(wrapped_obs.obs)
        self.obs = tuple(new_obs)

    def __call__(self) -> Tuple[np.ndarray]:
        return self.obs
