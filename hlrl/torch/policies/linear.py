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

class LinearCatPolicy(LinearPolicy):
    """
    A linear policy that takes multiple inputs and concatenates them.
    """
    def __init__(
            self,
            input_sizes: Tuple[int, ...],
            out_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        A policy that takes in a multiple inputs.

        Args:
            input_sizes: The sizes of the network inputs to concatenate.
            out_n: The number of output nodes.
            hidden_size: The size of each hidden layer.
            num_layers: The number of layers.
            activation_fn: The activation function between each layer.
        """
        super().__init__(
            sum(input_sizes), out_n, hidden_size, num_layers, activation_fn
        )

    def forward(
            self,
            *inputs: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the policy output for the input.

        Args:
            inputs: The inputs to concatenate.

        Returns:
            The network output on the input.
        """
        lin_in = torch.cat(inputs, dim=-1)
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
            hidden_size // len(split_space), out_features, hidden_size,
            num_layers - 1, activation_fn
        )

        first_out_n = hidden_size if num_layers > 1 else out_features

        self.split_layer = SplitLayer(
            dense_features, split_space, first_out_n // len(split_space)
        )

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
        split_forward = self.split_layer(dense_input, split_input)

        linear = super().forward(split_forward)
        linear = linear.view((*dense_input.shape[:-1], -1))

        return linear
