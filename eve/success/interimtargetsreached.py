from .success import Success
from ..interimtarget import InterimTarget


class InterimTargetsReached(Success):
    def __init__(self, interim_target: InterimTarget) -> None:
        self.interim_target = interim_target
        self._n_initial_targets = None

    def reset(self, *args, **kwds) -> None:
        self._n_initial_targets = len(self.interim_target.all_coordinates)

    @property
    def success(self) -> float:
        n_targets_left = len(self.interim_target.all_coordinates)
        if n_targets_left == 1 and self.interim_target.reached:
            success = 1.0
        else:
            success = 1 - (n_targets_left / self._n_initial_targets)
        return success
