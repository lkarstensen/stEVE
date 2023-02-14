import numpy as np
from .imaging import Imaging, gym


class Dummy(Imaging):
    def __init__(self, *args, **kwargs) -> None:
        ...

    @property
    def image_space(self) -> gym.Space:
        gym.spaces.Box(1, 1, (1, 1), dtype=np.uint8)

    @property
    def x_ray_image(self) -> np.ndarray:
        return np.array([[1]], dtype=np.uint8)

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...
