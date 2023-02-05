from typing import List

from .reward import Reward


class Combination(Reward):
    def __init__(self, rewards: List[Reward]) -> None:
        self.rewards = rewards

    def reset(self, episode_nr: int = 0) -> None:
        for reward in self.rewards:
            reward.reset(episode_nr)
        self.reward = 0.0

    def step(self) -> None:
        cum_reward = 0
        for reward in self.rewards:
            reward.step()
            cum_reward += reward.reward
        self.reward = cum_reward
