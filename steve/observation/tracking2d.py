import numpy as np

from ..intervention import Intervention
from .observation import Observation, gym


class Tracking2D(Observation):
    def __init__(
        self,
        intervention: Intervention,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = "tracking2d",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.n_points = n_points
        self.resolution = resolution
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        low = self.intervention.fluoroscopy.tracking2d_space.low
        high = self.intervention.fluoroscopy.tracking2d_space.high
        low = np.tile(low, [self.n_points, 1])
        high = np.tile(high, [self.n_points, 1])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        tracking = self.intervention.fluoroscopy.tracking2d
        self.obs = self._evenly_distributed_tracking(tracking)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()

    def _evenly_distributed_tracking(self, tracking: np.ndarray) -> np.ndarray:
        tracking = list(tracking)
        tracking_state = [tracking[0]]
        if self.n_points > 1:
            acc_dist = 0.0
            for point, next_point in zip(tracking[:-1], tracking[1:]):
                if len(tracking_state) >= self.n_points or np.all(point == next_point):
                    break
                length = np.linalg.norm(next_point - point)
                dist_to_point = self.resolution - acc_dist
                acc_dist += length
                while (
                    acc_dist >= self.resolution and len(tracking_state) < self.n_points
                ):
                    unit_vector = (next_point - point) / length
                    tracking_point = point + unit_vector * dist_to_point
                    tracking_state.append(tracking_point)
                    acc_dist -= self.resolution

            while len(tracking_state) < self.n_points:
                tracking_state.append(tracking_state[-1])
        return np.array(tracking_state, dtype=np.float32)
