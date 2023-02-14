from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from .imaging import Imaging, gym
from ..intervention import Intervention


class Pillow(Imaging):
    def __init__(
        self,
        intervention: Intervention,
        image_size: Tuple[int, int],
    ) -> None:
        self.intervention = intervention
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self._image_mode = "L"
        self.low = 0
        self.high = 255

        self._tracking_to_image_factor = 0
        self._tracking_offset = [0, 0]
        self._image_offset = [0, 0]
        self._image = None

    @property
    def image_space(self) -> gym.Space:
        return gym.spaces.Box(0, 255, self.image_size, dtype=np.uint8)

    @property
    def x_ray_image(self) -> np.ndarray:
        return self._image

    def step(self) -> np.ndarray:
        trackings = self.intervention.device_trackings
        diameters = self.intervention.device_diameters
        # Noise is around colour 128.
        noise_image = Image.effect_noise(size=self.image_size, sigma=5)
        physics_image = self._render(trackings, diameters)
        image = ImageChops.darker(physics_image, noise_image)
        self._image = np.asarray(image, dtype=np.uint8)

    def reset(self, episode_nr: int = 0) -> None:
        coords_high = self.intervention.tracking_space.high
        coords_low = self.intervention.tracking_space.low
        intervention_size_x = coords_high[0] - coords_low[0]
        intervention_size_y = coords_high[1] - coords_low[1]
        x_factor = self.image_size[0] / intervention_size_x
        y_factor = self.image_size[1] / intervention_size_y
        self._tracking_to_image_factor = min(x_factor, y_factor)
        self._tracking_offset = np.array([-coords_low[0], -coords_low[1]])
        x_image_offset = (
            self.image_size[0] - intervention_size_x * self._tracking_to_image_factor
        ) / 2
        y_image_offset = (
            self.image_size[1] - intervention_size_y * self._tracking_to_image_factor
        ) / 2
        self._image_offset = np.array([x_image_offset, y_image_offset])

    def _render(
        self, trackings: Dict[str, np.ndarray], diameters: Dict[str, float]
    ) -> None:
        physics_image = Image.new(
            mode=self._image_mode, size=self.image_size, color=255
        )

        lines = [
            [tracking, diameter]
            for tracking, diameter in zip(trackings.values(), diameters.values())
        ]

        lines = sorted(lines, key=lambda line: line[1])
        for i, line in enumerate(lines):
            diameter = int(np.round(line[1] * self._tracking_to_image_factor))
            coord_points = line[0]
            if i < len(trackings) - 1:
                end = coord_points.shape[0] - trackings[i + 1][0].shape[0]
            else:
                end = coord_points.shape[0] - 1
            self._draw_lines(
                physics_image,
                coord_points[:end],
                int(diameter),
                grey_value=40,
            )
        return physics_image

    def _draw_circle(
        self,
        image: np.ndarray,
        position: np.ndarray,
        radius: float,
        grey_value: int,
    ) -> np.ndarray:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        position = self._coord_transform_tracking_to_image(position)
        radius *= self._tracking_to_image_factor
        circle_bb_low = (coord - radius for coord in position)
        circle_bb_high = (coord + radius for coord in position)
        draw.ellipse([circle_bb_low, circle_bb_high], fill=grey_value)
        return np.asarray(image)

    def _draw_lines(
        self,
        image: Image,
        point_cloud: np.ndarray,
        width=1,
        grey_value=0,
    ) -> np.ndarray:

        draw = ImageDraw.Draw(image)
        point_cloud_image = self._coord_transform_tracking_to_image(point_cloud)
        draw.line(point_cloud_image, fill=grey_value, width=width, joint="curve")
        return np.asarray(image)

    def _coord_transform_tracking_to_image(
        self, coords: np.ndarray
    ) -> List[Tuple[float, float]]:

        coords_image = (coords + self._tracking_offset) * self._tracking_to_image_factor
        coords_image += self._image_offset
        coords_image = np.round(coords_image, decimals=0).astype(np.int64)
        coords_image[:, 1] = -coords_image[:, 1] + self.image_size[1]
        coords_image = [(coord[0], coord[1]) for coord in coords_image]
        return coords_image
