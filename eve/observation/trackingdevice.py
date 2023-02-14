import numpy as np

from ..intervention.intervention import Intervention, Device
from . import Tracking


class TrackingDevice(Tracking):
    def __init__(
        self,
        intervention: Intervention,
        device: Device,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = None,
    ) -> None:
        name = name or f"{device.name}_tracking"
        super().__init__(intervention, n_points, resolution, name)
        self.device = device

    def _calculate_tracking_state(self) -> np.ndarray:
        tracking = self.intervention.device_trackings[self.device]
        inserted_length = self.intervention.device_lengths_inserted[self.device]
        return self._evenly_distributed_tracking(tracking, inserted_length)
