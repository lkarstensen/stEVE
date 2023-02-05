from .imaging import Imaging, VesselTree, Intervention
from typing import Tuple, Optional
from PIL import Image, ImageChops
import numpy as np


class LNK1(Imaging):
    def __init__(
        self,
        intervention: Intervention,
        image_size: Tuple[int, int],
        dimension_to_omit="y",
        vessel_tree: Optional[VesselTree] = None,
    ) -> None:
        super().__init__(intervention, image_size, dimension_to_omit, vessel_tree)
        self._image = None

    @property
    def image(self) -> Image.Image:
        return self._image

    @property
    def pixel_bits(self) -> str:
        return 8

    @property
    def pixel_bands(self) -> str:
        return 1

    def step(self, inject_contrast_agent: float = 0.0):
        noise_image = Image.effect_noise(size=self.image_size, sigma=10)
        # Noise is around colour 110. Adjust with second image to move
        noise_image_adjustment = self.get_new_image(color=110)
        noise_image = ImageChops.add(noise_image, noise_image_adjustment)
        physics_image = self._render()
        image = ImageChops.darker(physics_image, noise_image)
        if inject_contrast_agent > 0:
            image = ImageChops.blend(image, self.vessel_tree_image, 0.5)
        self._image = image

    def close(self):
        ...

    def _render(self) -> None:
        physics_image = Image.new(mode="L", size=self.image_size, color=255)
        trackings = self.intervention.tracking_per_device
        for i in range(len(trackings)):
            trackings[i] = [trackings[i], self.intervention.device_diameters[i]]

        trackings = sorted(trackings, key=lambda tracking: tracking[1])
        for i, tracking in enumerate(trackings):
            diameter = int(np.round(tracking[1] * self.maze_to_image_factor))
            tracking = tracking[0]
            if i < len(trackings) - 1:
                end = tracking.shape[0] - trackings[i + 1][0].shape[0]
            else:
                end = tracking.shape[0] - 1
            for j in range(end):
                line_start = tracking[j]
                line_end = tracking[j + 1]
                self.draw_line(
                    physics_image,
                    line_start,
                    line_end,
                    int(diameter),
                    colour=70,
                )
        return physics_image
