import torch

from hlrl.core.common import MethodWrapper


class MunchausenAgent(MethodWrapper):
    """
    An agent using Munchausen-RL from:
    https://arxiv.org/pdf/2007.14430.pdf
    """
    def __init__(self, agent, alpha, temperature):
        """
        Turns the agent into a Munchausen agent.
        """
        super().__init__(agent)

        self.alpha = alpha
        self.temperature = temperature
        self.log_probs = 0

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        self.log_probs = algo_step[-1]

        return self.om.transform_algo_step(algo_step[:-1])

    def transform_reward(self, state, algo_step, reward, next_state):
        """
        Adds the Munchausen reward to the reward
        """
        return self.om.transform_reward(
            state, algo_step, reward + self.temperature * self.log_probs,
            next_state
        )
