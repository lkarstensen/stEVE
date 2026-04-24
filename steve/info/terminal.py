from typing import Dict, Any
from . import Info
from ..terminal import Terminal


class TerminalInfo(Info):
    def __init__(self, done: Terminal, name: str = "done") -> None:
        super().__init__(name)
        self.done = done

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.done.terminal}

