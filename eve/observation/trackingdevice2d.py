from ..intervention import Intervention
from .tracking2d import Tracking2D


class TrackingDevice2D(Tracking2D):
    def __init__(
        self,
        intervention: Intervention,
        device_idx: int,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = None,
    ) -> None:
        name = name or f"device_{device_idx}_tracking"
        super().__init__(intervention, n_points, resolution, name)
        self.device_idx = device_idx

    def step(self) -> None:
        tracking = self.intervention.fluoroscopy.device_trackings2d[self.device_idx]
        self.obs = self._evenly_distributed_tracking(tracking)
