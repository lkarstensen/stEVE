from .normalize import Normalize, Observation, Optional
import numpy as np


class NormalizeCustom(Normalize):
    def __init__(
        self,
        wrapped_obs: Observation,
        low: np.ndarray,
        high: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self.low
        high = self.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
