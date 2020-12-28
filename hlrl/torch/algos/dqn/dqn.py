from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from hlrl.core.logger import Logger
from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.common import polyak_average
from hlrl.torch.common.functional import initialize_weights

class DQN(TorchOffPolicyAlgo):
    """
    The DQN Algorithm https://arxiv.org/abs/1312.5602
    """
    def __init__(self,
                 q_func: nn.Module,
                 discount: float,
                 polyak: float,
                 target_update_interval: int,
                 q_optim: torch.optim.Optimizer,
                 device: str = "cpu",
                 logger: Optional[Logger] = None):
        """
        Creates the deep Q-network algorithm with the given parameters

        Args:
            q_func: The Q-function that takes in the observation and action.
            discount: The coefficient for the discounted values (0 < x < 1).
            polyak: The coefficient for polyak averaging (0 < x < 1).
            target_update_interval: The number of environment steps in between
                target updates.
            q_optim: The optimizer for the Q-function.
            device: The device of the tensors in the module.
            logger: The logger to log results while training and evaluating,
                default None.
        """
        super().__init__(device, logger)

        # All constants
        self._discount = discount
        self._polyak = polyak
        self._target_update_interval = target_update_interval
        self.q_optim_func = q_optim

        # The network
        self.q_func = q_func
        self.q_func_targ = deepcopy(q_func)

    def create_optimizers(self) -> None:
        """
        Creates the optimizers for the model.
        """
        self.q_optim = self.q_optim_func(self.q_func.parameters())

    def _get_q_and_target(self,
            states: torch.FloatTensor,
            actions: torch.LongTensor,
            rewards: torch.FloatTensor,
            next_states: torch.FloatTensor,
            terminals: torch.LongTensor
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the Q-val and target Q-value of the batch.
        
        Args:
            states: The states to recompute losses on.
            actions: The actions of the batch.
            rewards: The rewards of the batch.
            next_states: The next states of the batch.
            terminals: The terminal states of the batch.

        Returns:
            A tuple of the computed Q-value, and it's target.
        """
        q_vals = self.q_func(states)
        act_qs = q_vals.gather(1, actions)

        with torch.no_grad():
            next_qs = self.q_func_targ(next_states)
            next_q_max = next_qs.max(dim=-1, keepdim=True).values
            q_targ = rewards + (1 - terminals) * self._discount * next_q_max

        return act_qs, q_targ

    def _step_optimizers(self,
            states: torch.FloatTensor,
            actions: torch.LongTensor,
            rewards: torch.FloatTensor,
            next_states: torch.FloatTensor,
            terminals: torch.LongTensor
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

        Returns:
            A tuple of the newly computed Q-value, and it's target.
        """
        self.q_optim.step()

        # Get the new q value to update the experience replay
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func, self.q_func_targ, self._polyak)

        with torch.no_grad():
            return self._get_q_and_target(
                states, actions, rewards, next_states, terminals
            )

    def forward(self, observation: torch.FloatTensor, greedy: bool = False):
        """
        Get the model output for a batch of observations.

        Args:
            observation: A batch of observations from the environment.
            greedy: If true, always returns the action with the highest Q-value.

        Returns:
            The action and Q-values of all actions.
        """
        q_vals = self.q_func(observation)

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

        return action, q_vals

    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation: A batch of observations from the environment.

        Returns:
            The action and Q-value of the action.
        """
        with torch.no_grad():
            action, q_vals = self(observation)

        q_val = q_vals.gather(1, action)

        return action, q_val

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

        q_loss_func = nn.SmoothL1Loss(reduction='none')

        q_val, q_target = self._get_q_and_target(
            states, actions, rewards, next_states, terminals
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
            states, actions, rewards, next_states, terminals
        )

        self.training_steps += 1

        return new_qs, new_q_targ

