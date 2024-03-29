from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

from .linear import LinearPolicy

class GaussianPolicy(nn.Module):
    """
    A simple gaussian policy.
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
            self,
            inp_n: int,
            out_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        Creates the gaussian policy.

        Args:
            inp_n: The number of input units to the network.
            out_n: The number of output units from the network.
            hidden_size: The number of units in each hidden layer.
            num_layers: The number of layers before the gaussian layer.
            activation_fn: The activation function in between each layer.
        """
        super().__init__()

        self.linear = LinearPolicy(
            inp_n, hidden_size, hidden_size, num_layers - 1, activation_fn
        )

        if num_layers > 1:
            self.linear = nn.Sequential(self.linear, activation_fn())
            last_in_n = hidden_size
        else:
            last_in_n = inp_n

        self.mean = nn.Linear(last_in_n, out_n)
        self.log_std = nn.Linear(last_in_n, out_n)

    def compute_mean_and_std(
            self,
            inp: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log std of the policy on the input.

        Args:
            inp: The input tensor to put through the network.

        Returns:
            A mean and standard deviation of the gaussian distribution.
        """
        lin = self.linear(inp)
        mean = self.mean(lin)

        log_std = self.log_std(lin)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX
        )

        return mean, log_std

    def forward(
            self,
            inp: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.
            deterministic: If the means should be used rather than a sample.

        Returns:
            A gaussian distribution of the network.
        """
        mean, log_std = self.compute_mean_and_std(inp)
        std = log_std.exp()

        normal = Normal(mean, std)

        if deterministic:
            action = normal.mean
        else:
            action = normal.rsample()

        log_prob = normal.log_prob(action)

        return action, log_prob, mean

class TanhGaussianPolicy(GaussianPolicy):
    """
    A gaussian policy with an extra tanh layer (restricted to (-1, 1))
    """
    def __init__(
            self,
            inp_n: int,
            out_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module,
            action_range: int = 1
        ):
        """
        Creates the gaussian policy.

        Args:
            inp_n: The number of input units to the network.
            out_n: The number of output units from the network.
            hidden_size: The number of units in each hidden layer.
            num_layers: The number of layers before the gaussian layer.
            activation_fn: The activation function in between each layer.
            action_range: The range of the output (-action_range, action_range)
        """
        super().__init__(inp_n, out_n, hidden_size, num_layers, activation_fn)

        self.action_range = action_range

    def forward(
            self,
            inp: torch.Tensor,
            deterministic: bool = False,
            epsilon: int = 1e-4
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.
            deterministic: If the means should be used rather than a sample.
            epsilon: The value to add to the standard deviation to prevent
                division by zero.

        Returns:
            A tanh squished gaussian distribution of the network.
        """
        # Need to get mean before tanh
        sample, log_prob, mean = super().forward(inp)
        action = torch.tanh(sample)

        log_prob -= torch.log(self.action_range * (1 - action.pow(2)) + epsilon)

        action = action * self.action_range
        mean = torch.tanh(mean) * self.action_range

        return action, log_prob, mean


class MultiCategoricalPolicy(nn.Module):
    """
    A simple multi-categorical policy.
    """
    def __init__(
            self,
            inp_n: int,
            out_n: int,
            classes_n: int,
            hidden_size: int,
            num_layers: int,
            activation_fn: nn.Module
        ):
        """
        Creates the gaussian policy.

        Args:
            inp_n: The number of input units to the network.
            out_n: The number of independent categorical distributions.
            classes_n: The number of classes per category.
            hidden_size: The number of units in each hidden layer.
            num_layers: The number of layers before the gaussian layer.
            activation_fn: The activation function in between each layer.
        """
        super().__init__()

        self.num_classes = classes_n

        self.linear = LinearPolicy(
            inp_n, hidden_size, hidden_size, num_layers - 1, activation_fn
        )

        last_in_n = inp_n

        if num_layers > 1:
            self.linear = nn.Sequential(self.linear, activation_fn())
            last_in_n = hidden_size

        # Will reshape to (b, out_n, classes_n)
        self.value = nn.Linear(last_in_n, out_n * classes_n)
        self.probs = nn.Softmax(dim=-1)

    def forward(
            self,
            inp: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, None]:

        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.categorie

        Returns:
            A multi-categorical distribution of the network.
        """
        linear = self.linear(inp)

        value = self.value(linear)
        value = value.view(*value.shape[:-1], -1, self.num_classes)

        probs = self.probs(value)
        dist = OneHotCategorical(probs)

        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        # Straight through gradient trick
        sample = sample + probs - probs.detach()

        mean = torch.argmax(probs, dim=-1).values

        return sample, log_prob, mean
