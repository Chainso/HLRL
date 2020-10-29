import torch

from hlrl.core.agents import RecurrentAgent


class TorchRecurrentAgent(RecurrentAgent):
    """
    A recurrent agent that holds a hidden state.
    """
    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        # Transposing here to go (batch, 2, num layers, hidden size)
        transed_state = super().transform_state(state)
        transed_state["hidden_state"] = torch.stack(
            transed_state["hidden_state"]
        )

        return transed_state
