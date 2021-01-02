import torch
import torch.nn as nn
import math

from copy import deepcopy
from itertools import chain

from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.common import polyak_average

from hlrl.core.logger import TensorboardLogger

class RainbowIQN(TorchOffPolicyAlgo):
    """
    Implicit Quantile Networks with the rainbow of improvements used for DQN.
    https://arxiv.org/pdf/1908.04683.pdf (IQN)
    """
    def __init__(self, enc_dim, autoencoder, q_func, discount, polyak,
        n_quantiles, embedding_dim, huber_threshold, target_update_interval,
        enc_optim, q_optim, device="cpu", logger=None):
        """
        Creates the Rainbow-IQN network.

        Args:
            enc_dim (int) : The dimension of the encoded state tensor.
            autoencoder (torch.nn.Module) : The state autoencoder before fed
                                            into the quantile network.
            q_func (torch.nn.Module) : The Q-function that takes in the
                                       observation and action.
            discount (float) : The coefficient for the discounted values
                                (0 < x < 1).
            polyak (float) : The coefficient for polyak averaging (0 < x < 1).
            n_quantiles (int) : The number of quantiles to sample.
            embedding_dim (int) : The dimension of the embedding tensor.
            huber_threshold (float) : The huber loss threshold constant (kappa).
            target_update_interval (int): The number of training steps in
                                          between target updates.
            enc_optim (torch.nn.Module) : The optimizer for the autoencoder.
            q_optim (torch.nn.Module) : The optimizer for the Q-function.
            device (str): The device of the tensors in the module.
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super().__init__(device, logger)

        # Constants
        self.discount = discount
        self.polyak = polyak
        self.n_quantiles = n_quantiles
        self.embedding_dim = embedding_dim
        self.huber_threshold = huber_threshold
        self.target_update_interval = target_update_interval
        self.q_optim_func = q_optim
        self.enc_optim_func = enc_optim

        # Networks
        self.autoencoder = autoencoder

        self.q_func = q_func
        self.target_q_func = deepcopy(q_func)

        # Quantile layer
        self.quantiles = nn.Linear(self.embedding_dim, enc_dim)
        nn.init.uniform_(self.quantiles.weight)

        self.relu = nn.ReLU()

        self.action = nn.Softmax(-1)

    def create_optimizers(self):
        self.q_optim = self.q_optim_func(self.q_func.parameters())

        self.enc_optim = self.enc_optim_func(
            chain(self.autoencoder.parameters(), self.quantiles.parameters())
        )

    def _calculate_quantile_values(self, observation, q_func):
        """
        Calculates the quantile distribtion for the observations.

        Args:
            observation (torch.FloatTensor) : A batch of observations from the
                                              environment.
            q_func (torch.nn.Module) : The Q-function to use for the quantiles.
        """
        latent = self.autoencoder(observation)

        # Tile the feature dimension to the quantile dimension size
        latent_tiled = latent.repeat(self.n_quantiles, 1)

        # Sample a random number and tile for the embedding dimension
        quantiles = torch.rand(
            self.n_quantiles * observation.shape[0], 1,
            device=observation.device
        )
        quantiles_tiled = quantiles.repeat(1, self.embedding_dim)

        # RELU(sum_{i = 1}^{n} (cos(pi * i * sample) * w_ij + b_j))
        # No bias in this calculation however
        # Paper also has {i = 0}^{n - 1}, but is inconsistent with loss function
        quantile_values = torch.arange(
            1, self.embedding_dim + 1, device=observation.device
        )
        quantile_values = quantile_values * math.pi * quantiles_tiled
        quantile_values = self.relu(self.quantiles(torch.cos(quantile_values)))

        # Multiple with input feature dim
        quantile_values = latent_tiled * quantiles
        quantile_values = q_func(quantile_values)
  
        return quantile_values, quantiles

    def forward(self, observation, greedy=False):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor) : A batch of observations from the
                                              environment.
            greedy (boolean) : If true, always returns the action with the
                               highest Q-value.

        Returns:
            The action and Q-value.
        """
        quantile_values, _ = self._calculate_quantile_values(
            observation, self.q_func
        )

        quantile_values = quantile_values.view(
            self.n_quantiles, observation.shape[0], -1
        )

        # Get the mean to find the q values
        q_val = torch.mean(quantile_values, dim=0)

        probs = self.action(q_val)

        if self.logger is not None and observation.shape[0] == 1:
            with torch.no_grad():
                action_gap = torch.topk(probs, 2).values
                action_gap = action_gap[:, 0] - action_gap[:, 1]
                action_gap = action_gap.item()

                self.logger["Training/Action-Gap"] = (
                    action_gap, self.env_steps
                )


        if greedy:
            action = torch.argmax(probs, dim=1, keepdim=True)
        else:
            action = torch.multinomial(probs, 1)

        return action, q_val, probs

    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation (torch.FloatTensor): A single observation from the
                                             environment.

        Returns:
            The action and Q-value of the action.
        """
        with torch.no_grad():
            action, q_val, probs  = self(observation)

        q_val = q_val.gather(1, action)
        #log_probs = torch.clamp(torch.log(probs.gather(1, action)), -1, 0)

        return action, q_val

    def train_batch(self, rollouts, is_weights=1):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t) of training data for the
                               network.
            is_weights (numpy.array) : The importance sampling weights for PER.
        """
        # Get all the parameters from the rollouts
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]

        # Tile parameters for the quantiles
        actions = actions.repeat(self.n_quantiles, 1)
        rewards = rewards.repeat(self.n_quantiles, 1)

        terminal_mask = (1 - terminals).repeat(self.n_quantiles, 1)

        with torch.no_grad():
            target_quantile_values = self._calculate_q_target(
                rewards, next_states, terminal_mask
            )
            target_quantile_values = target_quantile_values.transpose(0, 1)

        quantile_values, quantiles = self._calculate_quantile_values(
            states, self.q_func
        )
        quantile_values = quantile_values.gather(1, actions)
        quantile_values = quantile_values.view(
            self.n_quantiles, states.shape[0], 1
        )
        quantile_values = quantile_values.transpose(0, 1)

        bellman_error = target_quantile_values - quantile_values
        abs_bellman_error = torch.abs(bellman_error)

        # Huber loss has two cases abs(bellman_error) <= huber_threshold (kappa)
        # and abs(bellman_error) > huber_threshold
        # Case 1 (MSE)
        huber_loss1 = (
            (abs_bellman_error <= self.huber_threshold) * 0.5
            * (torch.pow(bellman_error, 2))
        )

        # Case 2 (kappa * (abs(bellman_error) - (kappa/2)))
        huber_loss2 = (
            (abs_bellman_error > self.huber_threshold) * self.huber_threshold
            * (abs_bellman_error - 0.5 * self.huber_threshold)
        )

        huber_loss = huber_loss1 + huber_loss2

        quantiles = quantiles.view(self.n_quantiles, states.shape[0], 1)
        quantiles = quantiles.transpose(0, 1)

        # Quantile regression loss
        # sum_{i = 1}^{N} sum_{j = 1}^{N'} abs(quantiles - (bellman_error < 0))
        #                                  * (huber_loss / huber_threshold)
        qr_loss = (
            torch.abs(quantiles - (bellman_error < 0).float().detach())
            * huber_loss / self.huber_threshold
        )

        # Sum over embedding dimension
        qr_loss = qr_loss.sum(dim=2)

        # Mean over number of quantiles
        qr_loss = qr_loss.mean(dim=1)

        # Importance sampling weights
        qr_loss = qr_loss * is_weights
        qr_loss = qr_loss.mean(dim=0)

        # Backpropogate losses
        self.enc_optim.zero_grad()
        self.q_optim.zero_grad()

        qr_loss.backward()

        self.enc_optim.step()
        self.q_optim.step()

        # Calculate the new Q-values and target for PER
        with torch.no_grad():
            _, new_q_val = self.step(states)

            new_quantile_values_target = self._calculate_q_target(
                rewards, next_states, terminal_mask
            )
            new_q_target = torch.mean(new_quantile_values_target, dim=0)

        # Update the target
        if (self.training_steps % self.target_update_interval == 0):
            polyak_average(self.q_func, self.target_q_func, self.polyak)

        self.training_steps += 1

        # Log the loss
        if (self.logger is not None):
            self.logger["Train/QR Loss"] = (
                qr_loss.detach().item(), self.training_steps
            )

        return new_q_val, new_q_target

    def _calculate_q_target(self, rewards, next_states, terminal_mask):
        """
        Calculates the target Q-value, assumes tensors have been tiled for the
        quantiles already.

        Args:
            rewards (torch.Tensor) : The rewards of the batch.
            next_states (torch.Tensor) : The next states of the batch.
            terminal_mask (torch.Tensor) : A mask to remove terminal Q-value
                                           predictions.
        """
        # Get the online actions of the next states
        next_actions, _, _ = self(next_states, True)
        next_actions = next_actions.repeat(self.n_quantiles, 1)

        # Target network quantile values
        next_quantile_values, _ = self._calculate_quantile_values(
            next_states, self.target_q_func
        )

        # Use the greedy action to get target quantiles
        next_quantile_values = next_quantile_values.gather(1, next_actions)

        target_quantile_values = (
            rewards + terminal_mask * self.discount * next_quantile_values
        )
        
        target_quantile_values = target_quantile_values.view(
            self.n_quantiles, next_states.shape[0], 1
        )

        return target_quantile_values