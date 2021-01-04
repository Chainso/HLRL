import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from hlrl.torch.common import polyak_average
from hlrl.core.logger import Logger

from .iqn import RainbowIQN

class RainbowIQNRecurrent(RainbowIQN):
    """
    Implicit Quantile Networks with the rainbow of improvements used for DQN,
    using recurrent networks.
    https://arxiv.org/pdf/1908.04683.pdf (IQN)
    """
    def _calculate_quantile_values(
            self,
            observation: torch.Tensor,
            q_func: nn.Module,
            hidden_states: Any,
        ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Calculates the quantile distribtion for the observations.

        Args:
            observation: A batch of observations from the environment.
            q_func: The Q-function to use to calculte the quantiles.
            hidden_states: The recurrent hidden state.

        Returns:
            A tuple of sampled quantiles and the quantile distribution, and the
            next hidden state.
        """
        latent, next_hiddens = self.autoencoder(observation, hidden_states)

        # Tile the feature dimension to the quantile dimension size
        latent_tiled = latent.repeat(self.n_quantiles, 1, 1)
        latent_tiled = latent.view(
            observation.shape[0] * self.n_quantiles * observation.shape[1], -1
        )

        # Sample a random number and tile for the embedding dimension
        quantiles = torch.rand(
            self.n_quantiles * observation.shape[0] * observation.shape[1], 1,
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
        quantile_values = latent_tiled * quantiles_values
        quantile_values = q_func(quantile_values)
  
        return quantile_values, quantiles, next_hiddens

    def forward(
            self,
            observation: torch.FloatTensor,
            hidden_states: Any,
            greedy: bool = False
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Any]:
        """
        Get the model output for a batch of observations

        Args:
            observation: A batch of observations from the environment.
            greedy: If true, always returns the action with the highest Q-value.
            hidden_states: The hidden state for the observation.

        Returns:
            The action, Q-value and next hidden state
        """
        quantile_values, _, next_hiddens = self._calculate_quantile_values(
            observation, self.q_func, hidden_states
        )

        quantile_values = quantile_values.view(
            self.n_quantiles, observation.shape[0], observation.shape[1], -1
        )

        # Get the mean to find the q values
        q_val = torch.mean(quantile_values, dim=0)

        probs = self.action(q_val)

        if self.logger is not None and observation.shape[:2] == [1, 1]:
            with torch.no_grad():
                action_gap = torch.topk(probs[0], 2).values
                action_gap = action_gap[:, 0] - action_gap[:, 1]
                action_gap = action_gap.item()

                self.logger["Training/Action-Gap"] = (
                    action_gap, self.env_steps
                )


        if greedy:
            action = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            action = torch.multinomial(probs, 1)

        return action, q_val, next_hiddens

    def step(
            self,
            observation: torch.FloatTensor,
            hidden_states: Any,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Any]:
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation: A single observation from the environment.
            hidden_states: The hidden states for the observation.

        Returns:
            The action, Q-value of the action and next hidden states.
        """
        with torch.no_grad():
            action, q_val, next_hiddens  = self(observation, hidden_states)

        q_val = q_val.gather(-1, action)
        #log_probs = torch.clamp(torch.log(probs.gather(-1, action)), -1, 0)

        return action, q_val, next_hiddens

    def train_batch(
            self,
            rollouts: Dict[str, torch.Tensor],
            is_weights: Union[int, torch.Tensor] = 1
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals, hidden state, next hidden state) rollouts.

        Args:
            rollouts: The dict of (s, a, r, s', t, h, nh) training data for the
                network.
            is_weights: The importance sampling weights for PER.

        Returns:
            The updated Q-value and target Q-value.
        """
        # Get all the parameters from the rollouts
        states = rollouts["state"]
        actions = rollouts["action"].long()
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        hidden_states = rollouts["hidden_state"]
        next_hiddens = rollouts["next_hidden_state"]

        # Tile parameters for the quantiles
        actions = actions.repeat(self.n_quantiles, 1)
        rewards = rewards.repeat(self.n_quantiles, 1)

        terminal_mask = (1 - terminals).repeat(self.n_quantiles, 1)

        with torch.no_grad():
            target_quantile_values = self._calculate_q_target(
                rewards, next_states, terminal_mask, next_hiddens
            )
            target_quantile_values = target_quantile_values.view(
                self.n_quantiles, next_states.shape[0] * next_states.shape[1], 1
            )
            target_quantile_values = target_quantile_values.transpose(0, 1)

        quantile_values, quantiles, _ = self._calculate_quantile_values(
            states, self.q_func, hidden_states
        )
        quantile_values = quantile_values.gather(-1, actions)
        quantile_values = quantile_values.view(
            self.n_quantiles, states.shape[0] * states.shape[1], 1
        )
        quantile_values = quantile_values.transpose(0, 1)

        bellman_error = target_quantile_values - quantile_values
        abs_bellman_error = torch.abs(bellman_error)

        huber_loss = self.huber_loss(quantile_values, target_quantile_values)

        quantiles = quantiles.view(
            self.n_quantiles, states.shape[0] * states.shape[1], 1
        )
        quantiles = quantiles.transpose(0, 1)

        # Quantile regression loss
        # sum_{i = 1}^{N} sum_{j = 1}^{N'} abs(quantiles - (bellman_error < 0))
        #                                  * (huber_loss / huber_threshold)
        qr_loss = (
            torch.abs(quantiles - (bellman_error < 0).float().detach())
            * huber_loss / self.huber_threshold
        )

        # Sum over embedding dimension
        qr_loss = qr_loss.sum(dim=-1)

        # Mean over number of quantiles
        qr_loss = qr_loss.mean(dim=1)

        # Importance sampling weights
        qr_loss = qr_loss.view(states.shape[0], states.shape[1], -1)
        qr_loss = qr_loss * is_weights
        qr_loss = qr_loss.mean()

        # Backpropogate losses
        self.enc_optim.zero_grad()
        self.q_optim.zero_grad()

        qr_loss.backward()

        self.enc_optim.step()
        self.q_optim.step()

        # Calculate the new Q-values and target for PER
        with torch.no_grad():
            _, new_q_val, new_next_hiddens = self.step(states)

            new_quantile_values_target = self._calculate_q_target(
                rewards, next_states, terminal_mask, new_next_hiddens
            )
            new_quantile_values_target = new_target_quantile_values.view(
                self.n_quantiles, next_states.shape[0], next_states.shape[1], 1
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

    def _calculate_q_target(
            self,
            rewards: torch.FloatTensor,
            next_states: torch.FloatTensor,
            terminal_mask: torch.Tensor,
            next_hidden_states: Tuple[torch.FloatTensor, torch.FloatTensor]
        ) -> torch.FloatTensor:
        """
        Calculates the target Q-value, assumes tensors have been tiled for the
        quantiles already.

        Args:
            rewards: The rewards of the batch.
            next_states: The next states of the batch.
            terminal_mask: A mask to remove terminal Q-value predictions.
            next_hidden_states: The hidden states of the next states.

        Returns:
            The target Q-value.
        """
        rewards = rewards.view(
            rewards.shape[0] * rewards.shape[1], -1
        )

        # Get the online actions of the next states
        next_actions = self(next_states, next_hidden_states, True)[0]
        next_actions = next_actions.repeat(self.n_quantiles, 1, 1)
        next_actions = next_actions.view(
            next_actions.shape[0] * next_actions.shape[1], -1
        )

        # Target network quantile values
        next_quantile_values = self._calculate_quantile_values(
            next_states, self.target_q_func, next_hidden_states
        )[0]

        # Use the greedy action to get target quantiles
        next_quantile_values = next_quantile_values.gather(-1, next_actions)

        target_quantile_values = (
            rewards + terminal_mask * self.discount * next_quantile_values
        )

        return target_quantile_values
