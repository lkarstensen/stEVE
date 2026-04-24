import importlib
from .visualisation import Visualisation
from ..observation import Image


class FromState(Visualisation):
    def __init__(self, image_state: Image) -> None:
        self._matplotlib = importlib.import_module("matplotlib")
        self._matplotlib.use("TkAgg")
        self._plt = importlib.import_module("matplotlib.pyplot")

        self.fig, self.ax = self._plt.subplots()
        # self.ax.set_aspect("equal")
        # self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.image_state = image_state
        self._plt.pause(0.1)

    def render(self):
        self.ax.clear()
        image = self.image_state.image
        self.ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.000001)

    def close(self) -> None:
        self._plt.close(self.fig)

    def reset(self, episode_nr: int = 0) -> None:
        ...
