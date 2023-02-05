from .visualisation import Visualisation
from ..imaging import Imaging
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class FromImaging(Visualisation):
    def __init__(
        self,
        imaging: Imaging,
    ) -> None:
        self.imaging = imaging
        self.fig, self.ax = plt.subplots()
        # self.ax.set_aspect("equal")
        # self.ax.set_axis_off()
        self.fig.canvas.draw()
        plt.pause(0.1)

    def step(self):
        self.ax.clear()
        image = self.imaging.image
        self.ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.000001)

    def close(self) -> None:
        plt.close(self.fig)

    def reset(self, episode_nr: int = 0) -> None:
        ...
