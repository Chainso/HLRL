from abc import abstractmethod
from typing import Any

from hlrl.core.algos import RLAlgo
from hlrl.core.common.wrappers import MethodWrapper

class IntrinsicRewardAlgo(MethodWrapper):
    """
    A generic instrinsic reward algorithm.
    """
    def __init__(self, algo: RLAlgo):
        """
        Creates the wrapper with the instrinsic reward method to be overridden.

        Args:
            algo (RLAlgo): The algorithm to wrap.
        """
        super().__init__(algo)

    @abstractmethod
    def intrinsic_reward(self, state: Any, algo_step: Any, reward: Any,
        next_state: Any):
        """
        Computes the intrinsic reward of the states.

        Args:
            state (Any): The state of the environment.
            action (Any): The last action taken in the environment.
            reward (Any): The external reward to add to.
            next_state (Any): The new state of the environment.
        """
        raise NotImplementedError