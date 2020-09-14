import torch
import torch.nn as nn

class LinearPolicy(nn.Module):
    """
    A standard linear policy
    """
    def __init__(self,
        inp_n: int,
        out_n: int,
        hidden_size: int,
        num_layers: int,
        activation_fn: nn.Module):
        """
        Creates a network of layers of the given sizes with the activation
        function for each layer. The output layer has no activation function.

        Args:
            inp_n (int): The number of input nodes.
            out_n (int): The number of output nodes.
            hidden_size (int): The size of each hidden layer.
            num_layers (int): The number of layers.
            activation_fn (torch.nn.Module): The activation function between
                each layer.
        """
        super().__init__()

        if num_layers > 1:
            first_layer = self._lin_block(inp_n, hidden_size, activation_fn)
            last_in_n = hidden_size
        else:
            first_layer = nn.Identity()
            last_in_n = inp_n

        hidden_layers = [
            self._lin_block(hidden_size, hidden_size, activation_fn)
            for _ in range(num_layers - 2)
        ]

        last_layer = (
            nn.Linear(last_in_n, out_n) if num_layers > 0 else nn.Identity()
        )

        self.linear = nn.Sequential(first_layer, *hidden_layers, last_layer)

    def _lin_block(self, num_in, num_out, activation_fn):
        """
        Creates a linear block consisting of a linear layer and the activation
        function
        """
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            activation_fn()
        )

    def forward(self, inp):
        """
        Returns the policy output for the input
        """
        return self.linear(inp)


class LinearSAPolicy(LinearPolicy):
    """
    A linear policy that takes state-action inputs (e.g. continuous Q-policy)
    """
    def __init__(self,
        state_n: int,
        act_n: int,
        out_n: int,
        hidden_size: int,
        num_layers: int,
        activation_fn: nn.Module):
        """
        A policy that takes in a state and action, appending the action to the
        state.

        Args:
            state_n (int): The number of input nodes for the state.
            act_n (int): The number of input nodes for the action.
            out_n (int): The number of output nodes.
            hidden_size (int): The size of each hidden layer.
            num_layers (int): The number of layers.
            activation_fn (torch.nn.Module): The activation function between
                each layer.
        """
        super().__init__(
            state_n + act_n, out_n, hidden_size, num_layers, activation_fn
        )

    def forward(self, state, actions):
        lin_in = torch.cat([state, actions], dim=-1)
        return super().forward(lin_in)