import numpy as np
import gymnasium as gym

from .interimtarget import InterimTarget, Target
from ..pathfinder import Pathfinder
from ..intervention.intervention import Intervention


class Even(InterimTarget):
    def __init__(
        self,
        pathfinder: Pathfinder,
        intervention: Intervention,
        target: Target,
        resolution: float,
        threshold: float,
    ) -> None:
        super().__init__(intervention, threshold)
        self.pathfinder = pathfinder
        self.target = target
        self.resolution = resolution

    @property
    def coordinate_space2d(self) -> gym.spaces.Box:
        return self.pathfinder.coordinate_space

    def step(self) -> None:
        position = self.intervention.instrument_position_vessel_cs[0]
        position_to_target = self.all_coordinates[0] - position
        dist = np.linalg.norm(position_to_target)
        if dist < self.threshold:
            self.reached = True
            if len(self.all_coordinates) > 1:
                self.all_coordinates = self.all_coordinates[1:]
        else:
            self.reached = False
        self.coordinates2d = self.all_coordinates[0]

    def reset(self, episode_nr: int = 0) -> None:
        self.all_coordinates = self._calc_interim_targets()

    def _calc_interim_targets(self) -> np.ndarray:
        path_points = self.pathfinder.path_points
        path_points = path_points[::-1]
        interim_targets = [self.target.coordinates2d]
        acc_dist = 0.0
        for point, next_point in zip(path_points[:-1], path_points[1:]):
            length = np.linalg.norm(next_point - point)
            acc_dist += length
            while acc_dist >= self.resolution:
                unit_vector = (next_point - point) / length
                interim_target = next_point - unit_vector * (acc_dist - self.resolution)
                interim_targets.append(interim_target)
                acc_dist -= self.resolution

        interim_targets = interim_targets[::-1]
        return interim_targets
