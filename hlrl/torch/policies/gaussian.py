import torch
import torch.nn as nn

from torch.distributions import Normal

from .linear import LinearPolicy

class GaussianPolicy(nn.Module):
    """
    A simple gaussian policy
    """
    def __init__(self, inp_n, out_n, hidden_size, num_hidden, activation_fn):
        super().__init__()

        self.linear = nn.Sequential(
            LinearPolicy(inp_n, hidden_size, hidden_size, num_hidden - 1,
                         activation_fn),
            activation_fn()
        )

        self.mean = nn.Linear(hidden_size, out_n)
        self.log_std = nn.Linear(hidden_size, out_n)

    def forward(self, inp):
        """
        Returns the mean and log std of the policy on the input.
        """
        lin = self.linear(inp)
        mean = self.mean(lin)
        log_std = self.log_std(lin)

        return mean, log_std

    def sample(self, inp):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample
        """
        epsilon = 1e-4

        mean, log_std = self(inp)
        std = log_std.exp()

        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action)

        return action, mean, log_prob

class TanhGaussianPolicy(GaussianPolicy):
    """
    A gaussian policy with an extra tanh layer (restricted to [-1, 1])
    """
    def __init__(self, inp_n, out_n, hidden_size, num_hidden, activation_fn):
        super().__init__(inp_n, out_n, hidden_size, num_hidden, activation_fn)

        self.linear = nn.Sequential(
            LinearPolicy(inp_n, hidden_size, hidden_size, num_hidden - 1,
                         activation_fn),
            activation_fn()
        )

        self.mean = nn.Linear(hidden_size, out_n)
        self.log_std = nn.Linear(hidden_size, out_n)

    def forward(self, inp):
        """
        Returns the mean and log std of the policy on the input.
        """
        mean, log_std = super().forward(inp)
        mean = torch.tanh(mean)

        return mean, log_std

    def sample(self, inp):
        """
        Returns a sample of the policy on the input with the mean and log
        probability of the sample
        """
        epsilon = 1e-4

        # Need to get mean before tanh
        mean, log_std = super().forward(inp)
        std = log_std.exp()

        normal = Normal(mean, std)
        sample = normal.rsample()
        action = torch.tanh(sample)

        log_prob = normal.log_prob(sample) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean)

        return action, mean, log_prob