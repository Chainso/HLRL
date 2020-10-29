import torch

from typing import Any, Tuple, OrderedDict

from hlrl.core.common.wrappers import MethodWrapper


class RecurrentAgent(MethodWrapper):
    """
    A recurrent agent that holds a hidden state.
    """
    def __init__(self, agent):
        """
        Turns the agent into a recurrent agent.
        """
        super().__init__(agent)

        self.hidden_state = None

    def set_hidden_state(self, hidden_state):
        """
        Sets the hidden state to the given one.
        """
        self.hidden_state = hidden_state

    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        transed_state = self.om.transform_state(state)
        transed_state["hidden_state"] = self.hidden_state

        return transed_state

    def transform_algo_step(self,
                            algo_step: Tuple[Any]) -> OrderedDict[str, Any]:
        """
        Updates the hidden state to the last output of the algorithm.
        
        Args:
            algo_step: The outputs of the algorithm on the input state.

        Returns:
            An ordered dictionary of the algorithm outputs with
            "next_hidden_state" mapping to the returned hidden state.
        """
        transed_algo_step = super().transform_algo_step(algo_step[:-1])
        transed_algo_step["next_hidden_state"]: algo_step[-1]

        return transed_algo_step

    def transform_next_algo_step(self, next_algo_step):
        """
        Retrieves the next Q-value and discards the next hidden state.
        """
        transed_nas = self.om.transform_next_algo_step(next_algo_step)
        transed_nas.pop("next_next_hidden_state")

        return transed_nas

    def reset(self):
        """
        Resets the agent's hidden state
        """
        self.set_hidden_state(self.obj.algo.reset_hidden_state())