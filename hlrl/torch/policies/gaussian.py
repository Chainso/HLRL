import torch
import torch.nn as nn

from torch.distributions import Normal

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
            inp_n (int): The number of input units to the network.
            out_n (int): The number of output units from the network.
            hidden_size (int): The number of units in each hidden layer.
            num_layers (int): The number of layers before the gaussian layer.
            activation_fn (nn.Module): The activation function in between each
                layer.
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
        probability of the sample
        """
        mean, log_std = self.compute_mean_and_std(inp)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action)

        return action, mean, log_prob

class TanhGaussianPolicy(GaussianPolicy):
    """
    A gaussian policy with an extra tanh layer (restricted to [-1, 1])
    """
    def compute_mean_and_std(self, inp):
        """
        Returns the mean and log std of the policy on the input.
        """
        mean, log_std = super().compute_mean_and_std(inp)
        mean = torch.tanh(mean)

        return mean, log_std

    def forward(self, inp, epsilon=1e-4):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample
        """
        # Need to get mean before tanh
        sample, mean, log_prob = super().forward(inp)
        action = torch.tanh(sample)

        log_prob = log_prob - torch.log(1 - action.pow(2) + epsilon)

        # Not sure if need keep dim here
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean)

        return action, log_prob, mean