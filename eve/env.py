from copy import deepcopy
from importlib import import_module
from typing import List, Tuple, Dict, Any, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym
import pickle

from .intervention import Intervention
from .observation import Observation, ObsDict, ObsTuple
from .reward import Reward
from .terminal import Terminal
from .truncation import Truncation, TruncationDummy
from .info import Info, InfoDummy
from .util import EveObject, ConfigHandler
from .visualisation import Visualisation, VisualisationDummy
from .start import Start, InsertionPoint
from .pathfinder import Pathfinder, PathfinderDummy
from .interimtarget import InterimTarget, InterimTargetDummy

ObsType = TypeVar(
    "ObsType",
    np.ndarray,
    List[np.ndarray],
    Dict[str, np.ndarray],
)
RenderFrame = TypeVar("RenderFrame")


class Env(gym.Env, EveObject):
    def __init__(
        self,
        intervention: Intervention,
        observation: Union[Observation, ObsDict, ObsTuple],
        reward: Reward,
        terminal: Terminal,
        truncation: Optional[Truncation],
        info: Optional[Info] = None,
        start: Optional[Start] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:
        self.intervention = intervention
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.truncation = truncation or TruncationDummy()
        self.info = info or InfoDummy()
        self.start = start or InsertionPoint(intervention)
        self.visualisation = visualisation or VisualisationDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()

        self.episode_number = 0
        self._intervention_states = []

    @property
    def observation_space(self) -> gym.Space:
        return self.observation.space

    @property
    def action_space(self) -> gym.Space:
        return self.intervention.action_space

    def step(
        self,
        action: np.ndarray,
        store_intervention_state: bool = False,
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        self.intervention.step(action)
        self.pathfinder.step()
        self.interim_target.step()
        self.observation.step()
        self.reward.step()
        self.terminal.step()
        self.truncation.step()
        self.info.step()
        if store_intervention_state:
            self._intervention_states.append(self.intervention.get_step_state())
        return (
            deepcopy(self.observation()),
            deepcopy(self.reward.reward),
            deepcopy(self.terminal.terminal),
            deepcopy(self.truncation.truncated),
            deepcopy(self.info.info),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        store_intervention_state: bool = False,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.intervention.reset(self.episode_number, seed, options)
        self.start.reset(self.episode_number)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.reward.reset(self.episode_number)
        self.terminal.reset(self.episode_number)
        self.truncation.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        self._intervention_states = []
        if store_intervention_state:
            self._intervention_states.append(self.intervention.get_reset_state())
        return (
            deepcopy(self.observation()),
            deepcopy(self.info()),
        )

    def render(self) -> Optional[np.ndarray]:
        return self.visualisation.render()

    def close(self):
        self.intervention.close()
        self.visualisation.close()

    def save_intervention_states(self, path: str):
        with open(path, "wb") as handle:
            pickle.dump(
                self._intervention_states, handle, protocol=pickle.HIGHEST_PROTOCOL
            )


class EnvObsInfoOnly(Env):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        intervention: Intervention,
        observation: Union[Observation, ObsDict, ObsTuple],
        info: Optional[Info] = None,
        start: Optional[Start] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:
        self.intervention = intervention
        self.observation = observation
        self.info = info or InfoDummy()
        self.start = start or InsertionPoint(intervention)
        self.visualisation = visualisation or VisualisationDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()

        self.episode_number = 0
        self._intervention_states = []

    def step(
        self,
        action: np.ndarray,
        store_intervention_state: bool = False,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        self.intervention.step(action)
        self.pathfinder.step()
        self.interim_target.step()
        self.observation.step()
        self.info.step()
        if store_intervention_state:
            self._intervention_states.append(self.intervention.get_step_state())
        return (
            deepcopy(self.observation()),
            deepcopy(self.info.info),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        store_intervention_state: bool = False,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.intervention.reset(self.episode_number, seed, options)
        self.start.reset(self.episode_number)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        self._intervention_states = []
        if store_intervention_state:
            self._intervention_states.append(self.intervention.get_reset_state())
        return (
            deepcopy(self.observation()),
            deepcopy(self.info.info),
        )

    @classmethod
    def from_config_dict(cls, config_dict: Dict, to_exchange: Optional[Dict] = None):
        # Check if correct class in config dict
        config_class = config_dict["_class"]
        module_path, class_name = config_class.rsplit(".", 1)
        module = import_module(module_path)
        config_type = getattr(module, class_name)
        if not issubclass(config_type, Env):
            raise ValueError("Config File from wrong class")

        # reduce config dict to dependencies from info and obs
        new_config_dict = cls._get_reduced_config_dict(config_dict)

        return super().from_config_dict(new_config_dict, to_exchange)

    @classmethod
    def _get_reduced_config_dict(cls, config_dict):
        config_dict = deepcopy(config_dict)
        confighandler = ConfigHandler()
        # get registry of objects from full config
        obs_object_list = confighandler.config_dict_to_list_of_objects(
            config_dict["observation"], config_dict
        )
        info_object_list = confighandler.config_dict_to_list_of_objects(
            config_dict["info"], config_dict
        )
        reduced_object_list = obs_object_list
        reduced_object_list.update(info_object_list)

        # reduced config dict with only the ids from reduced_object_registry
        new_config_dict = {
            "_id": config_dict.pop("_id"),
            "_class": "eve.env.EnvObsInfoOnly",
        }
        config_dict.pop("_class")
        for entry, value in config_dict.items():
            if value["_id"] in reduced_object_list.keys():
                new_config_dict[entry] = value
        return new_config_dict
