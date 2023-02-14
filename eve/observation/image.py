from copy import deepcopy
import numpy as np
import PIL.Image

from .observation import Observation, gym

from ..imaging import Imaging

# TODO: Adjust to new imaging


class Image(Observation):
    def __init__(self, imaging: Imaging, name: str = "imaging") -> None:
        super().__init__(name)
        self.imaging = imaging
        self.image: PIL.Image.Image = None

    @property
    def space(self) -> gym.spaces.Box:
        high = np.ones(self.imaging.image_size, dtype=np.float32) * self.imaging.high
        low = np.ones(self.imaging.image_size, dtype=np.float32) * self.imaging.low
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.image = deepcopy(self.imaging.image)
        self.obs = np.array(self.image, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
