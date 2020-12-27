import torch

from typing import Any, OrderedDict

from hlrl.core.agents import RLAgent
from hlrl.core.algos import IntrinsicRewardAlgo
from hlrl.core.common.wrappers import MethodWrapper


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

        self.intrinsic_reward = 0

    def get_intrinsic_reward(self,
                             state: Any,
                             algo_step: OrderedDict[str, Any],
                             reward: Any,
                             terminal: Any,
                             next_state: Any) -> Any:
        """
        Returns the intrinsic reward of an experience tuple.

        Args:
            state: The state of the environment.
            algo_step: The transformed algorithm step of the state.
            reward: The reward from the environment.
            terminal: If the next state is a terminal state.
            next_state: The new state of the environment.

        Returns:
            The intrinsic reward of the experience.
        """
        self.intrinsic_reward = self.algo.intrinsic_reward(
            state, algo_step, reward, terminal, next_state
        )

        return self.intrinsic_reward

    def transform_reward(self,
                         state: Any,
                         algo_step: OrderedDict[str, Any],
                         reward: Any,
                         terminal: Any,
                         next_state: Any) -> Any:
        """
        Adds the intrinsic reward to the reward given.

        Args:
            state: The state of the environment.
            algo_step: The transformed algorithm step of the state.
            reward: The reward from the environment.
            terminal: If the next state is a terminal state.
            next_state: The new state of the environment.

        Returns:
            The intrinsic reward added with the reward.
        """
        self.get_intrinsic_reward(
            state, algo_step, reward, terminal, next_state
        )

        return self.om.transform_reward(
            state, algo_step, reward + self.intrinsic_reward, terminal,
            next_state
        )

    def reward_to_float(self,
                        reward: Any) -> float:
        """
        Subtracts back the intrinsic reward from the reward.

        Args:
            reward: The reward to turn into a float.

        Returns:
            The float value of the external reward.
        """
        return self.om.reward_to_float(reward - self.intrinsic_reward)