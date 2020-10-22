from typing import Any

from .intrinsic_reward import IntrinsicRewardAgent

class MunchausenAgent(IntrinsicRewardAgent):
    """
    An agent using Munchausen-RL from:
    https://arxiv.org/pdf/2007.14430.pdf
    """
    def __init__(self, agent, alpha):
        """
        Turns the agent into a Munchausen agent, assuming the underlying
        algorithm already has a temperature for their update
        """
        super().__init__(agent)

        self.alpha = alpha
        self.log_probs = 0

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        self.log_probs = algo_step[-1]

        return self.om.transform_algo_step(algo_step[:-1])

    def get_intrinsic_reward(self, state: Any, algo_step: Any, reward: Any,
        next_state: Any):
        """
        Returns the Munchausen reward on an experience tuple.

        Args:
            state (Any): The state of the environment.
            action (Any): The last action taken in the environment.
            reward (Any): The external reward to add to.
            next_state (Any): The new state of the environment.
        """
        self.intrinsic_reward = (
            self.alpha * self.algo.temperature * self.log_probs
        )

        return self.intrinsic_reward
