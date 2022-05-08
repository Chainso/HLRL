import torch

from hlrl.core.agents import RecurrentAgent


class TorchRecurrentAgent(RecurrentAgent):
    """
    A recurrent agent that holds a hidden state tensor.
    """
    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        transed_state = super().transform_state(state)

        if not isinstance(transed_state["hidden_state"], torch.Tensor):
            transed_state["hidden_state"] = torch.stack(
                transed_state["hidden_state"]
            )

        return transed_state
