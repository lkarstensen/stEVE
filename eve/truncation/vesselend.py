from .truncation import Truncation
from ..intervention import Intervention


class VesselEnd(Truncation):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    @property
    def truncated(self) -> bool:
        tip = self.intervention.simulation.dof_positions[0]
        return self.intervention.vessel_tree.at_tree_end(tip)

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
