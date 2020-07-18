import torch
import torch.nn as nn

from hlrl.torch.algos import TorchRLAlgo
from hlrl.torch.util import polyak_average

class RainbowIQN(TorchRLAlgo):
    """
    Implicit Quantile Networks with the rainbow of improvements used for DQN.
    https://arxiv.org/pdf/1908.04683.pdf (IQN)
    """
    def __init__(self, enc_dim, autoencoder, q_func, policy, discount,
        n_quantiles, embedding_dim, enc_optim, q_optim, p_optim, slogger=None):
        """
        Creates the Rainbow-IQN network.

        Args:
            enc_dim (int) : The dimension of the encoded state tensor.
            autoencoder (torch.nn.Module) : The state autoencoder before fed
                                            into the quantile network.
            q_func (torch.nn.Module) : The Q-function that takes in the
                                       observation and action.
            policy (torch.nn.Module) : The action policy that takes in the
                                       observation.
            discount (float) : The coefficient for the discounted values
                                (0 < x < 1).
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
            n_quantiles (int) : The number of quantiles to sample.
            embedding_dim (int) : The dimension of the embedding tensor.
            enc_optim (torch.nn.Module) : The optimizer for the autoencoder.
            q_optim (torch.nn.Module) : The optimizer for the Q-function.
            p_optim (torch.nn.Module) : The optimizer for the action policy.
        """
        super().__init__(logger)

        # Constants
        self.input_dim = input_dim
        self.discount = discount
        self.n_quantiles = n_quantiles
        self.embedding_dim = embedding_dim

        # Networks
        self.autoencoder = autoencoder
        self.enc_optim = enc_optim(self.autoencoder.parameters())

        self.q_func = q_func
        self.q_optim = q_optim(self.q_func.parameters())

        self.policy = policy
        self.p_optim = p_optim(self.policy.parameters())

        # Quantile layer
        self.quantiles = nn.Linear(self.embedding_dim, self.input_dim)
        nn.init.xavier_uniform_(self.quantiles)
        self.relu = nn.ReLU()

    def forward(self, observation):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.

        Returns:
            The action and Q-value.
        """
        latent = self.autoencoder(observation)

        # Tile the feature dimension to the quantile dimension size
        latent_tiled = latent.repeat(self.n_quantiles, 1)

        # Sample a random number and tile for the embedding dimension
        quantiles = torch.rand(self.n_quantiles * observation.shape[0], 1)
        quantiles = quantiles.to(observation.device)
        quantiles = quantiles.repeat(1, self.embedding_dim)

        # RELU(sum_{i = 0}^{n - 1} (cos(pi * i * sample) * w_ij + b_j))
        # No bias in this calculation however
        quantile_values = torch.range(0,
            self.embedding_dim).to(observation.device)
        quantile_values = quantile_values * torch.pi * quantiles
        quantile_values = self.relu(torch.cos(quantiles))

        # Multiple with input feature dim
        quantile_values = latent_tiled * quantiles
        quantile_values = self.q_func(quantile_values)

        # Get the mean to find the q values
        quantiles_values = quantile_values.view(self.n_quantiles,
            observation.state[0], -1)
        q_val = torch.mean(quantile_values, dim=0)

        return q_val, quantile_values, quantiles

    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation (torch.FloatTensor): A single observation from the
                                             environment.

        Returns:
            The action and Q-value of the action.
        """
        action, q_val = self(observation)

        return action.detach(), q_val.detach()