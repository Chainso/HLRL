import torch
import torch.nn as nn

from typing import Callable, Union, Tuple

class Conv2dPolicy(nn.Module):
    """
    A standard linear policy
    """
    def __init__(self,
        channel_sizes: Tuple[int, ...],
        kernel_sizes: Tuple[Union[int, Tuple[int, int]], ...],
        strides: Tuple[Union[int, Tuple[int, int]], ...],
        paddings: Tuple[Union[int, Tuple[int, int]], ...],
        activation_fn: nn.Module):
        """
        Creates a network of layers of the given sizes with the activation
        function for each layer. The output layer has no activation function.

        Args:
            channel_sizes (int): The size of each channel from input to output.
            kernel_sizes (Union[int, Tuple[int, int]]): The size of the
                convolution kernel.
            strides (Union[int, Tuple[int, int]]): The size of the stride.
            paddings (Union[int, Tuple[int, int]]): The size of the padding.
            activation_fn (nn.Module): The activation function in
                between each layer.
        """
        super().__init__()

        assert (
            len(channel_sizes) - 1 == len(kernel_sizes)
            == len(strides) == len(paddings)
        )

        layers = []
        for i in range(len(kernel_sizes) - 1):
            block = self._conv2d_block(
                channel_sizes[i], channel_sizes[i + 1], kernel_sizes[i],
                strides[i], paddings[i], activation_fn
            )
            layers.append(block)
 

        last_layer = (
            nn.Conv2d(
                channel_sizes[-2], channel_sizes[-1], kernel_sizes[-1],
                strides[-1], paddings[-1]
            )
            if len(kernel_sizes) > 0 else nn.Identity()
        )
    
        self.conv = nn.Sequential(*layers, last_layer)

    def _conv2d_block(self, inp_channels, out_channels, kernel_size, stride,
        padding, activation_fn):
        """
        Creates a linear block consisting of a linear layer and the activation
        function.

        Args:
            inp_channels (int): The number of input channels.
            out_channels (out): The number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): The size of the
                convolution kernel.
            stride (Union[int, Tuple[int, int]]): The size of the stride.
            padding (Union[int, Tuple[int, int]]): The size of the padding.
            num_layers (int): The number of layers in the network.
            activation_fn (nn.Module): The activation function in
                between each layer.
        """
        return nn.Sequential(
            nn.Conv2d(
                inp_channels, out_channels, kernel_size, stride, padding
            ),
            activation_fn()
        )

    def forward(self, inp):
        """
        Returns the policy output for the input
        """
        return self.conv(inp)