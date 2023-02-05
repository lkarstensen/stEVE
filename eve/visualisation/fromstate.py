from .visualisation import Visualisation
from .. import observation
from ..observation import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class FromState(Visualisation):
    def __init__(self, image_state: Image) -> None:
        self.fig, self.ax = plt.subplots()
        # self.ax.set_aspect("equal")
        # self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.image_state = image_state
        plt.pause(0.1)

    def step(self):
        self.ax.clear()
        image = self.image_state.image
        self.ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.000001)

    def close(self) -> None:
        plt.close(self.fig)

    def reset(self, episode_nr: int = 0) -> None:
        ...
