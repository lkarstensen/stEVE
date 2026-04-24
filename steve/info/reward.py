from typing import Dict, Any
from . import Info
from ..reward import Reward


class RewardInfo(Info):
    def __init__(self, reward: Reward, name: str = "reward") -> None:
        super().__init__(name)
        self.reward = reward

    @property
    def info(self) -> Dict[str, Any]:
        return {self.name: self.reward.reward}
