import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLayer(nn.Module):
    """
    A layer that provides noise to an input. Largely taken from:
    https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb
    """
    def __init__(self, input_size, output_size, std_init):
        """
        Initializes the noisy layer with the sizes provided and the initial
        standard deviation of the normal distribution provided.

        Args:
            input_size (int):   The number of input nodes.
            output_size (int):  The number of output nodes.
            std_init (int):     The standard deviation of the normal
                                distribution.
        """
        self.input_size = input_size
        self.output_size = output_size

        self.std_init = std

        # Weight parameters
        self.weight_mean = nn.Parameter(torch.empty(output_size, input_size))
        self.weight_std = nn.Parameter(torch.empty(output_size, input_size))
        self.register_buffer('weight_epsilon', torch.empty(
            output_size, input_size
        ))

        # Bias parameters
        self.bias_mean = nn.Parameter(torch.empty(output_size))
        self.bias_std = nn.Parameter(torch.empty(output_size))
        self.register_buffer('bias_epsilon', torch.empty(output_size))

        # Reset on init
        self.reset_parameters()
        self.reset_noise()

    def forward(self, inp):
        """
        Adds the noise to the input if training, otherwise does nothing.
        """
        
        if self.training:
            weight = self.weight_mean + self.weight_std * self.weight_epsilon
            bias = self.bias_mean + self.bias_std * self.bias_epsilon
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        out = F.linear(inp, weight, bias)

        return out

    def reset_parameters(self):
        """
        Resets the standard deviation to the initial one.
        """
        sqrt_input_size = np.sqrt(self.input_size)
        sqrt_output_size = np.sqrt(self.output_size)

        mean_range = 1.0 / sqrt_input_size

        self.weight_mean.data.uniform_(-mean_range, mean_range)
        self.weight_std.data.fill_(self.std_init / sqrt_input_size)
        self.bias_mean.data.uniform_(-mean_range, mean_range)
        self.bias_std.data.fill_(self.std_init / sqrt_output_size)

    def _scale_noise(self, size):
        """
        Returns noise generated from the standard normal distribution.

        Args:
            size (tuple | int): The shape of the noise.
        """
        noise = torch.randn(shape)
        scaled_noise = noise.sign().mul_(noise.abs().sqrt_())

        return scaled_noise

    def reset_noise(self):
        """
        Resets the noise of the layer.
        """
        epsilon_in = self._scale_noise(self.input_size)
        epsilon_out = self._scale_noise(self.output_size)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
