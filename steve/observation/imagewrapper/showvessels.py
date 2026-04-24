from typing import Dict, Optional
from PIL import ImageChops
import numpy as np

from .. import Image as ImageState
from ...intervention import Intervention


class ShowVessels(ImageState):
    def __init__(
        self,
        intervention: Intervention,
        wrapped_image: ImageState,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_image.name
        super().__init__(wrapped_image.intervention, name)
        self.intervention = intervention
        self.wrapped_image = wrapped_image
        self._overlay_image = None

    @property
    def space(self) -> Dict[str, np.ndarray]:
        return self.wrapped_image.space

    def step(self) -> None:
        self.wrapped_image.step()
        self.image = ImageChops.blend(
            self.wrapped_image.image, self._overlay_image, 0.5
        )

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_image.reset(episode_nr)
        self._create_overlay_image()
        self.image = ImageChops.blend(
            self.wrapped_image.image, self._overlay_image, 0.5
        )

    def _create_overlay_image(self):
        self._overlay_image = self.intervention.fluoroscopy.get_new_image(color=255)
        for branch in self.intervention.values():
            for coord, radius in zip(branch.coordinates, branch.radii):
                self._overlay_image = self.intervention.draw_circle(
                    self._overlay_image, coord, radius, colour=170
                )
