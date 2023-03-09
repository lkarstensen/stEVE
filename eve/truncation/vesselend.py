from .truncation import Truncation
from ..intervention import Intervention
from ..vesseltree import VesselTree


class VesselEnd(Truncation):
    def __init__(self, intervention: Intervention, vessel_tree: VesselTree) -> None:
        self.intervention = intervention
        self.vessel_tree = vessel_tree

    @property
    def truncated(self) -> bool:
        tip = self.intervention.instrument_position_vessel_cs[0]
        return self.vessel_tree.at_tree_end(tip)

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
