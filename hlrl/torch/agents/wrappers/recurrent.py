import torch

from hlrl.core.agents import RecurrentAgent


class TorchRecurrentAgent(RecurrentAgent):
    """
    A recurrent agent that holds a hidden state.
    """
    def __init__(self, agent):
        """
        Turns the agent into a recurrent agent.
        """
        super().__init__(agent)

    def transform_state(self, state):
        """
        Appends the hidden state to the algorithm inputs.
        """
        transed_state = super().transform_state(state)
        transed_state["hidden_state"] = torch.stack(
            transed_state["hidden_state"]
        )
        print(transed_state["hidden_state"].shape)
        return transed_state
