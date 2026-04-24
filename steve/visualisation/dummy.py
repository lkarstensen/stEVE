from .visualisation import Visualisation


class Dummy(Visualisation):
    def __init__(self, *args, **kwds) -> None:  # pylint: disable=unused-argument
        ...

    def render(self):
        ...

    def close(self):
        ...

    def reset(self, episode_nr=0):
        ...
