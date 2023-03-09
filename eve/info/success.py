from typing import Dict, Any
from .info import Info
from ..success import Success


class SuccessInfo(Info):
    def __init__(self, success: Success, name: str = "success") -> None:
        super().__init__(name)
        self.success = success

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.success.success}
