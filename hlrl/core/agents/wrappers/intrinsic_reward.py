import torch

from typing import Any

from hlrl.core.agents import RLAgent
from hlrl.core.algos import IntrinsicRewardAlgo
from hlrl.core.common import MethodWrapper


class IntrinsicRewardAgent(MethodWrapper):
    """
    A intrisic reward agent that adds a reward to the existing reward.
    """
    def __init__(self, agent: RLAgent):
        """
        Creates the intrinsic reward agent using the algorithm given to
        calculate the intrinsic reward.

        Args:
            agent (RLAgent): The agent to wrap.
        """
        super().__init__(agent)

    def transform_reward(self, state: Any, algo_step: Any, reward: Any,
        next_state: Any):
        """
        Adds the intrinsic reward to the reward given.

        Args:
            state (Any): The state of the environment.
            action (Any): The last action taken in the environment.
            reward (Any): The external reward to add to.
            next_state (Any): The new state of the environment.
        """
        return self.om.transform_reward(
            state, algo_step, reward + self.intrinsic_reward(state), next_state
        )
