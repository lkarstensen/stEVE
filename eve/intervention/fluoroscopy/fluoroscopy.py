# pylint: disable=unused-argument
from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from ...util import EveObject
from ...util.coordtransform import (
    vessel_cs_to_tracking3d,
    tracking3d_to_2d,
)
from ..simulation import Simulation
from ..vesseltree import VesselTree


class Fluoroscopy(EveObject):
    def __init__(
        self,
        simulation: Simulation,
        vessel_tree: VesselTree,
        image_frequency: float = 7.5,
        image_rot_zx: Tuple[float, float] = (0.0, 0.0),
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.simulation = simulation
        self.vessel_tree = vessel_tree
        self.image_rot_zx = image_rot_zx
        self.image_frequency = image_frequency
        vessel_low = vessel_tree.coordinate_space.low
        vessel_high = vessel_tree.coordinate_space.high
        vessel_center = (vessel_high + vessel_low) / 2
        self.image_center = image_center or vessel_center.tolist()
        self.field_of_view = field_of_view

    @property
    def image_space(self) -> gym.Space:
        return gym.spaces.Box(1, 1, (1, 1), dtype=np.uint8)

    @property
    def image(self) -> np.ndarray:
        return np.array([[1]], dtype=np.uint8)

    @property
    def tracking3d_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        low = vessel_cs_to_tracking3d(
            low, self.image_rot_zx, self.image_center, self.field_of_view
        )
        high = vessel_cs_to_tracking3d(
            high, self.image_rot_zx, self.image_center, self.field_of_view
        )
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d_space_episode(self) -> gym.spaces.Box:
        coords = self.vessel_tree.centerline_coordinates
        coords = vessel_cs_to_tracking3d(
            coords, self.image_rot_zx, self.image_center, self.field_of_view
        )
        low = np.min(coords, axis=0)
        low -= 0.1 * np.abs(low)
        high = np.max(coords, axis=0)
        high += 0.1 * np.abs(high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        space_3d = self.tracking3d_space
        low = tracking3d_to_2d(space_3d.low)
        high = tracking3d_to_2d(space_3d.high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        space_3d = self.tracking3d_space_episode
        low = tracking3d_to_2d(space_3d.low)
        high = tracking3d_to_2d(space_3d.high)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d(self) -> np.ndarray:
        return vessel_cs_to_tracking3d(
            self.simulation.dof_positions,
            self.image_rot_zx,
            self.image_center,
            self.field_of_view,
        )

    @property
    def tracking2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.tracking3d)

    @property
    def device_trackings3d(self) -> List[np.ndarray]:
        position = self.tracking3d
        position = np.flip(position)
        point_diff = position[:-1] - position[1:]
        length_btw_points = np.linalg.norm(point_diff, axis=-1)
        cum_length = np.cumsum(length_btw_points)
        inserted_lengths = self.simulation.inserted_lengths

        d_lengths = np.array(inserted_lengths)
        n_devices = d_lengths.size
        n_dofs = cum_length.size
        d_lengths = np.broadcast_to(d_lengths, (n_dofs, n_devices)).transpose()
        cum_length = np.broadcast_to(cum_length, (n_devices, n_dofs))

        diff = np.abs(cum_length - d_lengths)
        idxs = np.argmin(diff, axis=1)

        trackings = [np.flip(position[: idx + 1]) for idx in idxs]
        return trackings

    @property
    def device_trackings2d(self) -> List[np.ndarray]:
        trackings_3d = self.device_trackings3d
        trackings_2d = [tracking3d_to_2d(tracking) for tracking in trackings_3d]
        return trackings_2d

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "image_frequency": self.image_frequency,
            "image_rot_zx": self.image_rot_zx,
            "image_center": self.image_center,
            "field_of_view": self.field_of_view,
            "image_space": self.image_space,
            "image": self.image,
            "tracking2d_space": self.tracking2d_space,
            "tracking2d_space_episode": self.tracking2d_space_episode,
            "tracking3d_space": self.tracking3d_space,
            "tracking3d_space_episode": self.tracking3d_space_episode,
            "tracking3d": self.tracking3d,
            "tracking2d": self.tracking2d,
            "device_trackings3d": self.device_trackings3d,
            "device_trackings2d": self.device_trackings2d,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "image_frequency": self.image_frequency,
            "image_rot_zx": self.image_rot_zx,
            "image_center": self.image_center,
            "field_of_view": self.field_of_view,
            "image": self.image,
            "tracking3d": self.tracking3d,
            "tracking2d": self.tracking2d,
            "device_trackings3d": self.device_trackings3d,
            "device_trackings2d": self.device_trackings2d,
        }
        return deepcopy(state)
