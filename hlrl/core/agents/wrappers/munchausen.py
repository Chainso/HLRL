from typing import Any, Tuple

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

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the munchausen agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (type(self), (self.obj, self.alpha))

    def transform_algo_step(self, algo_step):
        """
        Updates the hidden state to the last output of the algorithm extras.
        """
        self.log_probs = algo_step[-1]

        return self.om.transform_algo_step(algo_step[:-1])

    def get_intrinsic_reward(self,
                             state: Any,
                             algo_step: Any,
                             reward: Any,
                             terminal: Any,
                             next_state: Any) -> None:
        """
        Returns the Munchausen reward on an experience tuple.

        Args:
            state: The state of the environment.
            action: The last action taken in the environment.
            reward: The external reward to add to.
            terminal: If this is the last step of the episode.
            next_state: The new state of the environment.
        """
        self.intrinsic_reward = (
            self.alpha * self.algo.temperature * self.log_probs
        )

        return self.intrinsic_reward
