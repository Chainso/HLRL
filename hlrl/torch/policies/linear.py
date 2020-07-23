import torch
import torch.nn as nn

class LinearPolicy(nn.Module):
    """
    A standard linear policy
    """
    def __init__(self, inp_n, out_n, hidden_size, num_hidden, activation_fn):
        """
        Creates a network of layers of the given sizes with the activation
        function for each layer. The output layer has no activation function.

        Args:
            inp_n (int): The number of input nodes.
            out_n (out): The number of output nodes.
            hidden_size (int): The size of each hidden layer.
            num_hidden (int): The number of hidden layers.
            activation_fn (torch.nn.Module): The activation function between
                                             each layer.
        """
        super().__init__()

        if(num_hidden < 0):
            # Identity function
            self.linear = nn.Sequential()
        elif(num_hidden == 0):
            self.linear = nn.Linear(inp_n, out_n)
        else:
            self.linear = nn.Sequential(
                self._lin_block(inp_n, hidden_size, activation_fn),
                *[self._lin_block(hidden_size, hidden_size, activation_fn)
                  for _ in range(num_hidden)],
                nn.Linear(hidden_size, out_n)
            )

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
    def __init__(self, state_n, act_n, out_n, hidden_size, num_hidden,
        activation_fn):
        super().__init__(
            state_n + act_n, out_n, hidden_size, num_hidden, activation_fn
        )

    def forward(self, state, actions):
        lin_in = torch.cat([state, actions], dim=-1)
        return super().forward(lin_in)