from ..simulation2d import Wall
from ..intervention import Intervention

from PIL import Image, ImageDraw
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..util import EveObject
from ..vesseltree import VesselTree
import numpy as np


class Imaging(EveObject, ABC):
    def __init__(
        self,
        intervention: Intervention,
        image_size: Tuple[int, int],
        dimension_to_omit: str = "y",
        vessel_tree: Optional[VesselTree] = None,
    ) -> None:
        self.intervention = intervention
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.dimension_to_omit = dimension_to_omit
        self.vessel_tree = vessel_tree
        if self.dimension_to_omit == "x":
            self._dim_del = 0
        elif self.dimension_to_omit == "z":
            self._dim_del = 2
        else:
            self._dim_del = 1

    @property
    @abstractmethod
    def pixel_bits(self) -> str:
        ...

    @property
    @abstractmethod
    def pixel_bands(self) -> str:
        ...

    @property
    @abstractmethod
    def image(self) -> Image.Image:
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def step(self, inject_contrast_agent: float = 0.0) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        tracking_high = self.intervention.tracking_high
        tracking_low = self.intervention.tracking_low
        if tracking_high.shape[-1] == 3:
            tracking_high = np.delete(tracking_high, self._dim_del, axis=-1)
            tracking_low = np.delete(tracking_low, self._dim_del, axis=-1)
        maze_size_x = tracking_high[0] - tracking_low[0]
        maze_size_y = tracking_high[1] - tracking_low[1]
        x_factor = self.image_size[0] / (maze_size_x)
        y_factor = self.image_size[1] / (maze_size_y)
        self.maze_to_image_factor = min(x_factor, y_factor)
        self.x_maze_offset = -self.intervention.tracking_low[0]
        self.y_maze_offset = -self.intervention.tracking_low[1]
        self.x_image_offset = (
            self.image_size[0] - maze_size_x * self.maze_to_image_factor
        ) / 2
        self.y_image_offset = (
            self.image_size[1] - maze_size_y * self.maze_to_image_factor
        ) / 2
        self.step()
        if self.vessel_tree is not None:
            self._create_vessel_tree_image()

    def get_new_image(self, color: int):
        return Image.new(mode="L", size=self.image_size, color=color)

    def draw_circle(
        self,
        image: Image.Image,
        position: np.ndarray,
        radius: float,
        colour: int,
    ) -> Image.Image:
        # position and radius in maze coordinate system
        if position.shape[-1] == 3:
            position = np.delete(position, self._dim_del, axis=-1)
        draw = ImageDraw.Draw(image)
        position = self._coord_transform_maze_to_image(position)
        radius *= self.maze_to_image_factor
        circle_bb_low = tuple([coord - radius for coord in position])
        circle_bb_high = tuple([coord + radius for coord in position])
        draw.ellipse([circle_bb_low, circle_bb_high], fill=colour)
        return image

    def draw_line(
        self,
        image: Image.Image,
        start: np.ndarray,
        end: np.ndarray,
        width=1,
        colour=0,
    ):
        if start.shape[-1] == 3:
            start = np.delete(start, self._dim_del, axis=-1)
            end = np.delete(end, self._dim_del, axis=-1)

        draw = ImageDraw.Draw(image)
        start = self._coord_transform_maze_to_image(start)
        end = self._coord_transform_maze_to_image(end)
        draw.line([start, end], fill=colour, width=width)
        return image

    def draw_polygon(
        self,
        image: Image.Image,
        corners: List[np.ndarray],
        fill=None,
        outline=None,
    ):
        draw = ImageDraw.Draw(image)
        corners_image = []
        for corner in corners:
            if corner.shape[-1] == 3:
                corner = np.delete(corner, self._dim_del, axis=-1)
            corners_image.append(self._coord_transform_maze_to_image(corner))
        draw.polygon(corners_image, fill=fill, outline=outline)
        return image

    def _coord_transform_maze_to_image(
        self, coord: Tuple[float, float]
    ) -> Tuple[float, float]:
        x = coord[0]
        x_image = (x + self.x_maze_offset) * self.maze_to_image_factor
        x_image = x_image + self.x_image_offset
        x_image = int(np.round(x_image))

        y = coord[1]
        y_image = (y + self.y_maze_offset) * self.maze_to_image_factor
        y_image = y_image + self.y_image_offset
        y_image = self.image_size[1] - y_image
        y_image = int(np.round(y_image))
        return (x_image, y_image)

    def _vector_transform_maze_to_image(
        self, vector: Tuple[float, float]
    ) -> Tuple[float, float]:
        x = vector[0] * self.maze_to_image_factor

        y = -vector[1] * self.maze_to_image_factor

        return (x, y)

    def _get_unit_vector(self, vector: Tuple[float, float]) -> Tuple[float, float]:
        length = (vector[0] ** 2 + vector[1] ** 2) ** (1 / 2)
        unit_vector = [vector[0] / length, vector[1] / length]
        return tuple(unit_vector)

    def _create_vessel_tree_image(self):
        self.vessel_tree_image = self.get_new_image(color=110)
        for branch in self.vessel_tree.centerline_tree.branches:
            for point in branch.centerline_points:
                self.vessel_tree_image = self.draw_circle(
                    self.vessel_tree_image,
                    point.to_point().to_np_array(),
                    point.radius,
                    colour=170,
                )
