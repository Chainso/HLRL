from abc import abstractmethod

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
    def intrinsic_reward(self, states):
        """
        Computes the intrinsic reward of the states.
        """
        raise NotImplementedError