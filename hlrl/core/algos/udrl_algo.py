from abc import ABC, abstractmethod
from typing import Any, OrderedDict, NoReturn

class UDRLAlgo(ABC):
    """
    A UDRL algorithm.
    """
    @abstractmethod
    def choose_command(
            self,
            state: Any,
            algo_step: OrderedDict[str, Any],
            reward: Any,
            next_state: Any
        ) -> NoReturn:
        """
        Chooses a command for the agent.

        Args:
            state: The state of the environment.
            action: The last action taken in the environment.
            reward: The external reward to add to.
            next_state: The new state of the environment.

        Raises:
            NotImplementedError if not overriden.
        """
        raise NotImplementedError
