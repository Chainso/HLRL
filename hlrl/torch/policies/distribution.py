import torch
import torch.nn as nn

from torch.distributions import Normal, Categorical, OneHotCategorical

from .linear import LinearPolicy

class GaussianPolicy(nn.Module):
    """
    A simple gaussian policy.
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self,
        inp_n: int,
        out_n: int,
        hidden_size: int,
        num_layers: int,
        activation_fn: nn.Module):
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

        last_in_n = hidden_size if num_layers > 1 else inp_n

        self.mean = nn.Linear(last_in_n, out_n)
        self.log_std = nn.Linear(last_in_n, out_n)

    def compute_mean_and_std(self, inp):
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

    def forward(self, inp):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.

        Returns:
            A gaussian distribution of the network.
        """
        mean, log_std = self.compute_mean_and_std(inp)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.rsample()

        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

class TanhGaussianPolicy(GaussianPolicy):
    """
    A gaussian policy with an extra tanh layer (restricted to (-1, 1))
    """
    def __init__(self,
        inp_n: int,
        out_n: int,
        hidden_size: int,
        num_layers: int,
        activation_fn: nn.Module,
        action_range: int = 1):
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

    def forward(self, inp, epsilon=1e-4):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.

        Returns:
            A tanh squished gaussian distribution of the network.
        """
        # Need to get mean before tanh
        sample, log_prob, mean = super().forward(inp)
        action = torch.tanh(sample)

        log_prob -= torch.sum(
            torch.log(self.action_range * (1 - action.pow(2)) + epsilon),
            dim=-1, keepdim=True
        )

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

        if num_layers > 1:
            self.linear = nn.Sequential(self.linear, activation_fn())
            last_in_n = hidden_size
        else:
            last_in_n = inp_n

        last_in_n = hidden_size if num_layers > 1 else inp_n

        # Will reshape to (b, out_n, classes_n)
        self.probs = nn.Linear(last_in_n, out_n * classes_n)
        self.value = nn.Sequential(
            nn.Linear(last_in_n, out_n * classes_n),
            nn.Tanh()
        )

    def forward(self, inp: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample.

        Args:
            inp: The input tensor to put through the network.categorie

        Returns:
            A multi-categorical distribution of the network.
        """
        linear = self.linear(inp)

        probs = self.probs(linear)
        probs = probs.view(linear.shape[0], -1, self.num_classes)
        probs = nn.Softmax(dim=-1)(probs)

        value = self.value(linear)
        value = value.view(linear.shape[0], -1, self.num_classes)

        dist = OneHotCategorical(probs)

        # Straight through gradient trick
        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        sample = sample + probs - probs.detach()
        out = torch.sum(value * sample, dim=-1)

        return out, log_prob, None
