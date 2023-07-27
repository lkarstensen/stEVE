from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from ..vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy
from ...util.coordtransform import (
    vessel_cs_to_tracking3d,
    tracking3d_to_2d,
)


class FluoroscopyDummy(Fluoroscopy):
    def __init__(
        self,
        image_frequency: float,
        image_rot_zx: Tuple[float, float],
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
        image_space: Optional[gym.spaces.Box] = None,
        tracking3d_space: Optional[gym.spaces.Box] = None,
        tracking3d_space_episode: Optional[gym.spaces.Box] = None,
        tracking2d_space: Optional[gym.spaces.Box] = None,
        tracking2d_space_episode: Optional[gym.spaces.Box] = None,
    ) -> None:
        self.image_frequency = image_frequency
        self.image_rot_zx = image_rot_zx
        self.image_center = image_center or [0, 0, 0]
        self.field_of_view = field_of_view

        self.image_space = image_space
        self.tracking3d_space = tracking3d_space
        self.tracking3d_space_episode = tracking3d_space_episode
        self.tracking2d_space = tracking2d_space
        self.tracking2d_space_episode = tracking2d_space_episode

        self.image = None

        self.tracking3d = None
        self.tracking2d = None
        self.device_trackings3d = None
        self.device_trackings2d = None


class FluoroscopyDummyWithVesselTree(Fluoroscopy):
    def __init__(
        self,
        vessel_tree: VesselTree,
        image_frequency: float = 7.5,
        image_rot_zx: Optional[Tuple[float, float]] = None,
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
        image_space: Optional[gym.spaces.Box] = None,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.image_frequency = image_frequency
        self.image_rot_zx = image_rot_zx or [0, 0]
        self.image_center = image_center or [0,0,0]
        self.field_of_view = field_of_view

        self.image_space = image_space

        self.image = None

        self.tracking3d = None
        self.tracking2d = None
        self.device_trackings3d = None
        self.device_trackings2d = None

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
