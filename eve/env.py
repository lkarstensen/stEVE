from copy import deepcopy
from importlib import import_module
from typing import Tuple, Dict, Any, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym

from .interimtarget import InterimTarget, InterimTargetDummy
from .pathfinder import Pathfinder, PathfinderDummy
from .intervention import Intervention
from .target import Target
from .start import Start, StartDummy
from .visualisation import Visualisation, VisualisationDummy
from .vesseltree import VesselTree, VesselTreeDummy
from .observation import Observation, ObsDict, ObsTuple
from .reward import Reward
from .terminal import Terminal
from .truncation import Truncation, TruncationDummy
from .info import Info, InfoDummy
from .imaging import Imaging, ImagingDummy
from .util import EveObject, ConfigHandler

ObsType = TypeVar(
    "ObsType",
    np.ndarray,
    Tuple[Union[np.ndarray, Dict[str, np.ndarray]]],
    Dict[str, np.ndarray],
)
RenderFrame = TypeVar("RenderFrame")


class Env(gym.Env, EveObject):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        target: Target,
        start: Start,
        observation: Union[Observation, ObsDict, ObsTuple],
        reward: Reward,
        terminal: Terminal,
        truncation: Optional[Truncation],
        info: Optional[Info] = None,
        imaging: Optional[Imaging] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.intervention = intervention
        self.target = target
        self.start = start
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.truncation = truncation or TruncationDummy()
        self.info = info or InfoDummy()
        self.imaging = imaging or ImagingDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()
        self.visualisation = visualisation or VisualisationDummy()

        self.episode_number = 0

    @property
    def observation_space(self) -> gym.Space:
        return self.observation.space

    @property
    def action_space(self) -> gym.Space:
        return self.intervention.action_space

    def step(
        self, action: np.ndarray
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        self.vessel_tree.step()
        self.intervention.step(action)
        self.imaging.step()
        self.pathfinder.step()
        self.target.step()
        self.interim_target.step()
        self.observation.step()
        self.reward.step()
        self.terminal.step()
        self.truncation.step()
        self.info.step()
        return (
            self.observation(),
            self.reward.reward,
            self.terminal.terminal,
            self.truncation.truncated,
            self.info.info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(self.episode_number, vessel_seed)
        self.intervention.reset(self.episode_number)
        self.start.reset(self.episode_number)
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(self.episode_number, target_seed)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.imaging.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.reward.reset(self.episode_number)
        self.terminal.reset(self.episode_number)
        self.truncation.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        return self.observation()

    def render(self) -> Optional[np.ndarray]:
        return self.visualisation.render()

    def close(self):
        self.intervention.close()
        self.visualisation.close()


class EnvObsInfoOnly(Env):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        intervention: Intervention,
        target: Target,
        observation: Union[Observation, ObsDict, ObsTuple],
        info: Optional[Info] = None,
        vessel_tree: Optional[VesselTree] = None,
        start: Optional[Start] = None,
        imaging: Optional[Imaging] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:
        self.intervention = intervention
        self.target = target
        self.observation = observation
        self.info = info or InfoDummy()
        self.vessel_tree = vessel_tree or VesselTreeDummy()
        self.start = start or StartDummy()
        self.imaging = imaging or ImagingDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()
        self.visualisation = visualisation or VisualisationDummy()

        self.episode_number = 0

    def step(self, action: np.ndarray) -> Tuple[ObsType, Dict[str, Any]]:
        self.vessel_tree.step()
        self.intervention.step(action)
        self.imaging.step()
        self.pathfinder.step()
        self.target.step()
        self.interim_target.step()
        self.observation.step()
        self.info.step()
        return (
            self.observation(),
            self.info.info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        self._np_random = np.random.default_rng(seed=seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(self.episode_number, vessel_seed)
        self.intervention.reset(self.episode_number)
        self.start.reset(self.episode_number)
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(self.episode_number, target_seed)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.imaging.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        return self.observation()

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
