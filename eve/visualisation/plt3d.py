from .visualisation import Visualisation
from ..vesseltree import VesselTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d


##### Not functional yet


class PLT3D(Visualisation):
    def __init__(
        self,
        vessel_tree: VesselTree,
    ) -> None:
        self.vessel_tree = vessel_tree
        self._init_vessel_tree()

        self._click_coordinates = None

    def render(self, tracking, target):
        self._plot_target(target)
        self._plot_tracking(tracking)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.000001)

    def close(self):
        plt.close("all")

    def _init_vessel_tree(self):
        self._initialize_pyplot()

    def _initialize_pyplot(
        self,
    ) -> None:
        if hasattr(self, "fig"):
            self.fig.clear()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection="3d")
        margins = [
            (
                self.vessel_tree.coordinates_high_local[0]
                - self.vessel_tree.coordinates_low_local[0]
            ),
            (
                self.vessel_tree.coordinates_high_local[1]
                - self.vessel_tree.coordinates_low_local[1]
            ),
            (
                self.vessel_tree.coordinates_high_local[2]
                - self.vessel_tree.coordinates_low_local[2]
            ),
        ]
        margin = max(margins)
        self.ax.set_xlim3d(
            self.vessel_tree.coordinates_low_local[0],
            self.vessel_tree.coordinates_low_local[0] + margin,
        )
        self.ax.set_ylim3d(
            self.vessel_tree.coordinates_low_local[1] - margin / 2,
            self.vessel_tree.coordinates_low_local[1] + margin / 2,
        )
        self.ax.set_zlim3d(
            self.vessel_tree.coordinates_low_local[2],
            self.vessel_tree.coordinates_low_local[2] + margin,
        )
        self.fig.canvas.mpl_connect("pick_event", self._on_click)
        for _, centerline in self.vessel_tree.centerline_point_cloud.items():
            centerline = np.delete(centerline, [3], axis=1)
            x = np.delete(centerline, [1, 2], axis=1).reshape(
                -1,
            )

            y = np.delete(centerline, [0, 2], axis=1).reshape(
                -1,
            )
            z = np.delete(centerline, [0, 1], axis=1).reshape(
                -1,
            )
            line = mplot3d.art3d.Line3D(
                x,
                y,
                z,
                picker=True,
                pickradius=3,
            )
            self.ax.add_artist(line)

        origin = self.vessel_tree.centerline_point_cloud_flat[0]
        x = origin[0]
        y = origin[1]
        z = origin[2]
        self._target_plot = mplot3d.art3d.Line3D(
            x,
            y,
            z,
            marker="X",
            label="Target",
            color="y",
            markersize=10,
        )
        self.ax.add_artist(self._target_plot)

        self._tracking_plot = mplot3d.art3d.Line3D(
            x,
            y,
            z,
            marker="o",
            label="Guidewire",
            color="g",
            markersize=8,
        )
        self.ax.add_artist(self._tracking_plot)
        self.fig.canvas.draw()
        plt.pause(0.001)
        self.fig.canvas.start_event_loop(0.00001)

    def _plot_tracking(self, tracking):
        x = tracking[:, 0]
        y = tracking[:, 1]
        z = tracking[:, 2]
        self._tracking_plot.set_data_3d(x, y, z)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.00001)

    def _plot_target(self, target):
        x = target[0]
        y = target[1]
        z = target[2]
        self._target_plot.set_data_3d(x, y, z)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.00001)

    def _on_click(self, event):
        data = event.artist.get_data_3d()
        idx = event.ind[0] + int((event.ind[-1] - event.ind[0]) / 2)
        x = data[0][idx]
        y = data[1][idx]
        z = data[2][idx]
        self._click_coordinates = np.array([x, y, z])
