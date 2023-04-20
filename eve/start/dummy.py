from .start import Start


class StartDummy(Start):
    def __init__(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        pass
