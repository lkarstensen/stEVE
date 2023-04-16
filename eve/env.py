from typing import Tuple, Dict, Any, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym

from .interimtarget import InterimTarget, InterimTargetDummy
from .pathfinder import Pathfinder, PathfinderDummy
from .intervention import Intervention
from .target import Target
from .start import Start
from .visualisation import Visualisation, VisualisationDummy
from .vesseltree import VesselTree
from .observation import Observation
from .reward import Reward
from .terminal import Terminal
from .truncation import Truncation, TruncationDummy
from .info import Info, InfoDummy
from .imaging import Imaging, ImagingDummy
from .util import EveObject

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
        observation: Observation,
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


class DummyEnv(Env):
    def __init__(  # pylint: disable=super-init-not-called
        self, *args, **kwds  # pylint: disable=unused-argument
    ) -> None:
        ...

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=np.empty((1,)), high=np.empty((1,)))

    def step(self, action: np.ndarray) -> None:
        ...

    def reset(self, *args, **kwds) -> None:  # pylint: disable=unused-argument
        ...

    def render(self) -> None:
        ...

    def close(self) -> None:
        ...
