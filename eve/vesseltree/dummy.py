import numpy as np
import gymnasium as gym

from .vesseltree import VesselTree


class VesselTreeDummy(VesselTree):
    def __init__(self) -> None:
        self.branches = None
        self.insertion = None
        self.branching_points = None
        self.centerline_coordinates = np.zeros((1, 3), dtype=np.float32)
        self.coordinate_space = gym.spaces.Box(0.0, 0.0, (3,))
        self.coordinate_space_episode = gym.spaces.Box(0.0, 0.0, (3,))
        self.mesh_path = None
        self.visu_mesh_path = None

    def step(self) -> None:
        ...

    def reset(self, episode_nr=0, seed: int = None) -> None:
        ...
