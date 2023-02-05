from abc import ABC
from typing import Dict

from .observation import Observation, gym, np


class ObsDict(ABC):
    def __init__(
        self, observations: Dict[str, Observation], name: str = "observation_dict"
    ) -> None:
        self.name = name
        self.obs: Dict[str, np.ndarray] = {}
        self.observations = observations

    @property
    def space(self) -> gym.spaces.Dict:
        spaces_dict = {name: obs.space for name, obs in self.observations.items()}
        return gym.spaces.Dict(spaces_dict)

    def step(self) -> None:
        for name, wrapped_obs in self.observations.items():
            wrapped_obs.step()
            self.obs[name] = wrapped_obs.obs

    def reset(self, episode_nr: int = 0) -> None:
        for name, wrapped_obs in self.observations.items():
            wrapped_obs.reset(episode_nr)
            self.obs[name] = wrapped_obs.obs

    def __call__(self) -> Dict[str, np.ndarray]:
        return self.obs
