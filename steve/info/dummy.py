from typing import Dict, Any
from . import Info


class Dummy(Info):
    def __init__(self, *args, **kwds) -> None:
        super().__init__("dummy")
        _ = args
        _ = kwds

    @property
    def info(self) -> Dict[str, Any]:
        return {}
