from typing import Tuple

import torch
import torch.nn as nn

from hlrl.torch.layers import SplitLayer
class LinearPolicy(nn.Module):
    """
    A standard linear policy.
    """
    def __init__(
            self,
            inp_n: int,
            out_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        Creates a network of layers of the given sizes with the activation
        function for each layer. The output layer has no activation function.

        Args:
            inp_n: The number of input nodes.
            out_n: The number of output nodes.
            hidden_size: The size of each hidden layer.
            num_layers: The number of layers.
            activation_fn: The activation function between each layer.
        """
        super().__init__()

        last_in_n = inp_n

        layers = []
        for _ in range(num_layers - 1):
            layers += list(self._lin_block(
                last_in_n, hidden_size, activation_fn
            ))

            last_in_n = hidden_size

        last_layer = (
            nn.Linear(last_in_n, out_n) if num_layers > 0 else nn.Identity()
        )

        self.linear = nn.Sequential(*layers, last_layer)

    def _lin_block(
            self,
            num_in: int,
            num_out: int,
            activation_fn: nn.Module
        ) -> nn.Module:
        """
        Creates a linear block consisting of a linear layer and the activation
        function.

        Args:
            num_in: The number of input units.
            num_out: The number of output units.
            activation_fn: The activation function after the linear layer.

        Returns:
            A module composed of the linear layer with the activation function.
        """
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            activation_fn()
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Returns the policy output for the input.

        Args:
            inp: The input to the network.

        Returns:
            The network output on the input.
        """
        return self.linear(inp)

class LinearSAPolicy(LinearPolicy):
    """
    A linear policy that takes state-action inputs (e.g. continuous Q-policy)
    """
    def __init__(
            self,
            state_n: int,
            act_n: int,
            out_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        A policy that takes in a state and action, appending the action to the
        state.

        Args:
            state_n: The number of input nodes for the state.
            act_n: The number of input nodes for the action.
            out_n: The number of output nodes.
            hidden_size: The size of each hidden layer.
            num_layers: The number of layers.
            activation_fn: The activation function between each layer.
        """
        super().__init__(
            state_n + act_n, out_n, hidden_size, num_layers, activation_fn
        )

    def forward(
            self,
            state: torch.Tensor,
            actions: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the policy output for the input.

        Args:
            state: The state of the transition.
            actions: The action to take at that state.

        Returns:
            The network output on the input.
        """
        lin_in = torch.cat([state, actions], dim=-1)
        return super().forward(lin_in)

class SplitLinearPolicy(LinearPolicy):
    """
    A policy starting with a split-linear layer.
    """
    def __init__(
            self,
            dense_features: int,
            split_space: Tuple[int, ...],
            out_features: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        A policy that takes in both dense and split features.

        Args:
            dense_features: The number of dense input features.
            split_space: The tuple space of the split input features.
            out_features: The number of output features per split input.
            hidden_size: The size of each hidden layer.
            num_layers: The number of layers.
            activation_fn: The activation function between each layer.
        """
        super().__init__(
            hidden_size, out_features, hidden_size, num_layers - 1,
            activation_fn
        )

        first_out_n = hidden_size if num_layers > 1 else out_features

        self.split_layer = SplitLayer(dense_features, split_space, first_out_n)

    def forward(
            self,
            dense_input: torch.Tensor,
            split_input: torch.Tensor
        ) -> torch.Tensor:
        """
        Does a forward pass on the concatenated and repeated dense input and
        split inputs.

        Args:
            dense_input: The inputs for the dense features.
            split_input: The inputs for the split features.

        Returns:
            The forward pass on the inputs.
        """
        batch_size = dense_input.shape[0]

        split_forward = self.split_layer(dense_input, split_input)

        linear = super().forward(split_forward)
        linear = linear.view(batch_size, -1)

        return linear
