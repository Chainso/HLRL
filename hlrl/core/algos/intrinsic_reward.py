from abc import ABC, abstractmethod
from typing import Any, OrderedDict, NoReturn

class IntrinsicRewardAlgo(ABC):
    """
    A generic instrinsic reward algorithm.
    """
    @abstractmethod
    def intrinsic_reward(self,
                         state: Any,
                         algo_step: OrderedDict[str, Any],
                         reward: Any,
                         next_state: Any) -> NoReturn:
        """
        Computes the intrinsic reward of the states.

        Args:
            state: The state of the environment.
            action: The last action taken in the environment.
            reward: The external reward to add to.
            next_state: The new state of the environment.

        Raises:
            NotImplementedError if not overriden.
        """
        raise NotImplementedError