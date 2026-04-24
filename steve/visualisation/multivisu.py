from typing import List
from .visualisation import Visualisation


class MultiVisualisation(Visualisation):

    def __init__(self, visualisations: List[Visualisation]):
        self.visualisations = visualisations

    def render(self) -> None:
        for visu in self.visualisations:
            visu.render()

    def reset(self, episode_nr: int = 0) -> None:
        for visu in self.visualisations:
            visu.reset()

    def close(self) -> None:
        for visu in self.visualisations:
            visu.close()
