from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dqn import DQN
from hlrl.core.logger import Logger
from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.common import polyak_average
from hlrl.torch.common.functional import initialize_weights

class DQNRecurrent(DQN):
    """
    A recurrent DQN Algorithm https://arxiv.org/abs/1312.5602
    """
    def calculate_q_and_target(self,
            states: torch.FloatTensor,
            actions: torch.LongTensor,
            rewards: torch.FloatTensor,
            next_states: torch.FloatTensor,
            terminals: torch.LongTensor,
            hidden_states: Tuple[torch.FloatTensor, torch.FloatTensor]
        ) -> Tuple[torch.FloatTensor,
                   torch.FloatTensor,
                   Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Computes the Q-val and target Q-value of the batch.
        
        Args:
            states: The states to recompute losses on.
            actions: The actions of the batch.
            rewards: The rewards of the batch.
            next_states: The next states of the batch.
            terminals: The terminal states of the batch.
            hidden_states: The hidden states of the batch.

        Returns:
            A tuple of the computed Q-value, its target and the next hidden
            states.
        """
        q_vals, next_hidden_states = self.q_func(states, hidden_states)
        act_qs = q_vals.gather(1, actions)

        with torch.no_grad():
            next_qs = self.q_func_targ(next_states, next_hidden_states)
            next_q_max = next_qs.max(dim=-1, keepdim=True).values
            q_targ = rewards + (1 - terminals) * self._discount * next_q_max

        return act_qs, q_targ, next_hidden_states

    def _step_optimizers(self,
            states: torch.FloatTensor,
            actions: torch.LongTensor,
            rewards: torch.FloatTensor,
            next_states: torch.FloatTensor,
            terminals: torch.LongTensor,
            hidden_states: Tuple[torch.FloatTensor, torch.FloatTensor]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Assumes the gradients have been computed and updates the parameters of
        the network with the optimizers.

        Args:
            states: The states to recompute losses on.
            actions: The actions of the batch.
            rewards: The rewards of the batch.
            next_states: The next states of the batch.
            terminals: The terminal states of the batch.
            hidden_states: The hidden states of the batch.

        Returns:
            A tuple of the newly computed Q-value, and its target.
        """
        self.q_optim.step()

        # Get the new q value to update the experience replay
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func, self.q_func_targ, self._polyak)

        with torch.no_grad():
            return self.calculate_q_and_target(
                states, actions, rewards, next_states, terminals, hidden_states
            )

    def forward(self,
            observation: torch.FloatTensor,
            hidden_state: Tuple[torch.FloatTensor, torch.FloatTensor],
            greedy: bool = False
        ) -> Tuple[torch.FloatTensor,
                   torch.FloatTensor,
                   Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Get the model output for a batch of observations.

        Args:
            observation: A batch of observations from the environment.
            hidden_state: The hidden states of the batch.
            greedy: If true, always returns the action with the highest Q-value.

        Returns:
            The action and Q-values of all actions with the next hidden state.
        """
        q_vals, next_hidden_state = self.q_func(observation, hidden_state)

        if self.logger is not None and observation.shape[0] == 1:
            with torch.no_grad():
                action_gap = torch.topk(q_vals, 2).values
                action_gap = action_gap[:, 0] - action_gap[:, 1]
                action_gap = action_gap.item()

                self.logger["Training/Action-Gap"] = (
                    action_gap, self.env_steps
                )

        probs = nn.Softmax(dim=-1)(q_vals)

        if greedy:
            action = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            action = torch.multinomial(probs, 1)

        return action, q_vals, next_hidden_state

    def step(self,
             observation: torch.FloatTensor,
             hidden_state: Tuple[torch.FloatTensor, torch.FloatTensor]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation: A batch of observations from the environment.
            hidden_state: The hidden states of the batch.

        Returns:
            The action and Q-value of the action.
        """
        with torch.no_grad():
            action, q_vals, next_hidden_state = self(observation)

        q_val = q_vals.gather(1, action)

        return action, q_val, next_hidden_state

    def train_batch(self,
                    rollouts: Dict[str, torch.Tensor],
                    is_weights: Union[int, torch.FloatTensor] = 1):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.
            is_weights (numpy.array) : The importance sampling weights for PER.
        """
        rollouts = {
            key: value.to(self.device) for key, value in rollouts.items()
        }

        # Get all the parameters from the rollouts
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        hidden_states = rollouts["hidden_state"]

        q_loss_func = nn.SmoothL1Loss(reduction='none')

        q_val, q_target = self.calculate_q_and_target(
            states, actions, rewards, next_states, terminals, hidden_states
        )

        q_loss = q_loss_func(q_val, q_target)
        q_loss = torch.mean(q_loss * is_weights)
        q_loss.backward()

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q Loss"] = (
                q_loss.detach().item(), self.training_steps
            )

        new_qs, new_q_targ = self._step_optimizers(
            states, actions, rewards, next_states, terminals, hidden_states
        )

        self.training_steps += 1

        return new_qs, new_q_targ
