from typing import Optional
import numpy as np

from .interimtarget import InterimTarget
from ..pathfinder import Pathfinder
from ..intervention import Intervention


class Even(InterimTarget):
    def __init__(
        self,
        pathfinder: Pathfinder,
        intervention: Intervention,
        resolution: float,
        threshold: float,
    ) -> None:
        self.intervention = intervention
        self.threshold = threshold
        self.pathfinder = pathfinder
        self.resolution = resolution
        # self.all_coordinates3d = []

    def step(self) -> None:
        self.reached = False
        position = self.intervention.fluoroscopy.tracking3d[0]
        if self.coordinates3d is not None:
            position_to_target = self.coordinates3d - position
            dist = np.linalg.norm(position_to_target)
            if dist < self.threshold:
                self.reached = True
                self.all_coordinates3d.pop(0)

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.all_coordinates3d = self._calc_interim_targets()

    def _calc_interim_targets(self) -> np.ndarray:
        path_points = self.pathfinder.path_points3d
        path_points = path_points[::-1]
        interim_targets = []
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
