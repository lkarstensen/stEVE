import numpy as np

from ..intervention import Intervention, gym
from . import Tracking


class TrackingAllDevices(Tracking):
    def __init__(
        self,
        intervention: Intervention,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = "tracking_all_devices",
    ) -> None:
        super().__init__(intervention, n_points, resolution, name)

    @property
    def space(self) -> gym.spaces.Box:
        n_devices = len(self.intervention.device_diameters)
        low = super().space.low
        high = super().space.high
        low = np.tile(low, [n_devices, 1, 1])
        high = np.tile(high, [n_devices, 1, 1])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _calculate_tracking_state(self) -> np.ndarray:
        trackings = self.intervention.tracking_per_device
        state_per_device = []
        for i, tracking in enumerate(trackings):
            inserted_length = self.intervention.device_lengths_inserted[i]
            tracking_state = self._evenly_distributed_tracking(
                tracking, inserted_length
            )
            state_per_device.append(tracking_state)
        return np.array(state_per_device, dtype=np.float32)
