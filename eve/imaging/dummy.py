from .imaging import Imaging, Image


class Dummy(Imaging):
    def __init__(self, *args, **kwargs) -> None:
        self.image_size = (1, 1)

    @property
    def image(self) -> Image.Image:
        return Image.new("L", (1, 1))

    @property
    def pixel_bits(self) -> str:
        1

    @property
    def pixel_bands(self) -> str:
        1

    def step(self, *args, **kwargs):
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...

    def close(self):
        ...
