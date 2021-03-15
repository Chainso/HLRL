from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from hlrl.core.logger import Logger
from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.common import polyak_average
from hlrl.torch.common.functional import initialize_weights

class SAC(TorchOffPolicyAlgo):
    """
    The Soft Actor-Critic algorithm from https://arxiv.org/abs/1801.01290
    """
    def __init__(
            self,
            action_space: Tuple,
            q_func: nn.Module,
            policy: nn.Module,
            discount: float,
            polyak: float,
            target_update_interval: int,
            q_optim: torch.optim.Optimizer,
            p_optim: torch.optim.Optimizer,
            temp_optim: torch.optim.Optimizer,
            twin: bool = True,
            device: Union[torch.device, str] = "cpu",
            logger: Optional[Logger] = None
        ):
        """
        Creates the soft actor-critic algorithm with the given parameters.

        Args:
            action_space: The dimensions of the action space of the environment.
            q_func: The Q-function that takes in the observation and action.
            policy: The action policy that takes in the observation.
            discount: The coefficient for the discounted values (0 < x < 1).
            polyak: The coefficient for polyak averaging (0 < x < 1).
            target_update_interval: The number of environment steps in between
                target updates.
            q_optim: The optimizer for the Q-function.
            p_optim: The optimizer for the action policy.
            temp_optim: The optimizer for the temperature.
            twin: If the twin Q-function algorithm should be used, default True.
            device: The device of the tensors in the module.
            logger: The logger to log results while training
                and evaluating, default None.
        """
        super().__init__(device, logger)

        # All constants
        self._discount = discount
        self._polyak = polyak
        self._target_update_interval = target_update_interval
        self.twin = twin
        self.q_optim_func = q_optim
        self.p_optim_func = p_optim
        self.temp_optim_func = temp_optim

        # The networks
        self.q_func1 = q_func

        with torch.no_grad():
            self.q_func_targ1 = deepcopy(self.q_func1)

        # Instantiate a second Q-function for twin SAC
        if(self.twin):
            self.q_func2 = deepcopy(q_func)
            self.q_func2.apply(initialize_weights(nn.init.xavier_uniform_))

            with torch.no_grad():
                self.q_func_targ2 = deepcopy(self.q_func2)

        self.q_loss_func = nn.MSELoss(reduction='none')

        self.policy = policy

        # Entropy tuning, starting at 1 due to auto-tuning
        self.temperature = 1
        self.target_entropy = -torch.prod(torch.Tensor(action_space)).item()
        self.log_temp = nn.Parameter(torch.zeros(1), requires_grad=True)

    def create_optimizers(self) -> None:
        """
        Creates the optimizers for the model.
        """
        self.q_optim1 = self.q_optim_func(self.q_func1.parameters())

        if self.twin:
            self.q_optim2 = self.q_optim_func(self.q_func2.parameters())

        self.p_optim = self.p_optim_func(self.policy.parameters())

        self.temp_optim = self.temp_optim_func([self.log_temp])

    def _step_optimizers(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assumes the gradients have been computed and updates the parameters of
        the network with the optimizers.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The updated Q-values and target Q-values.
        """
        self.q_optim1.step()

        if self.twin:
            self.q_optim2.step()

        self.p_optim.step()
        self.temp_optim.step()

        self.temperature = torch.exp(self.log_temp).item()

        with torch.no_grad():
            # Using Q-loss and zero as the target to make things a bit simpler
            q_loss = self.get_critic_loss(rollouts)

            if self.twin:
                q_loss = q_loss[0]

            zero_target = torch.zeros_like(q_loss)

        # Update the target
        if self.training_steps % self._target_update_interval == 0:
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)

            if self.twin:
                polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        return q_loss, zero_target

    def forward(
            self,
            observation: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the model output for a batch of observations

        Args:
            observation: A batch of observations from the environment.

        Returns:
            The action and Q-value.
        """
        action, _, _ = self.policy(observation)
        q_val = self.q_func1(observation, action)

        return action, q_val

    def step(
            self,
            observation: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the model action for a single observation of the environment.

        Args:
            observation: A single observation from the environment.

        Returns:
            The action and Q-value of the action.
        """
        with torch.no_grad():
            return self(observation)

    def get_critic_loss(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculates the loss for the Q-function/functions.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The batch-wise loss for the Q-function/functions.
        """
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy(next_states)
            next_log_probs = next_log_probs.sum(-1, keepdim=True)

            q_targ_pred = self.q_func_targ1(next_states, next_actions)

            if self.twin:
                q_targ_pred2 = self.q_func_targ2(next_states, next_actions)
                q_targ_pred = torch.min(q_targ_pred, q_targ_pred2)

            q_targ = q_targ_pred - self.temperature * next_log_probs
            q_next = rewards + (1 - terminals) * self._discount * q_targ

        q_pred = self.q_func1(states, actions)
        q_loss = self.q_loss_func(q_pred, q_next)

        if self.twin:
            q_pred2 = self.q_func2(states, actions)
            q_loss2 = self.q_loss_func(q_pred2, q_next)

            q_loss = (q_loss, q_loss2)
    
        return q_loss
        
    def get_actor_loss(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for the actor/policy.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The batch-wise loss for the actor/policy and the log probability of
            a sampled action on the current policy.
        """
        states = rollouts["state"]

        pred_actions, pred_log_probs, _ = self.policy(states)
        pred_log_probs = pred_log_probs.sum(-1, keepdim=True)
        
        p_q_pred = self.q_func1(states, pred_actions)

        if self.twin:
            p_q_pred2 = self.q_func2(states, pred_actions)
            p_q_pred = torch.min(p_q_pred, p_q_pred2)

        p_loss = self.temperature * pred_log_probs - p_q_pred

        return p_loss, pred_log_probs

    def get_entropy_loss(self, pred_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for entropy.

        Args:
            pred_log_probs: The log probabilities of actions of the current
                policy on the state.

        Returns:
            The loss for the entropy for soft Q-learning.
        """
        # Tune temperature
        with torch.no_grad():
            targ_entropy = pred_log_probs + self.target_entropy

        temp_loss = -self.log_temp * targ_entropy

        return temp_loss

    def train_batch(
            self,
            rollouts: Dict[str, torch.Tensor],
            is_weights: Union[int, torch.Tensor] = 1
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.
            is_weights: The importance sampling weights for PER.

        Returns:
            The updated Q-value and Q-value target.
        """
        # Make sure to change device if needed
        rollouts = {
            key: tens.to(self.device) for key, tens in rollouts.items()
        }

        if isinstance(is_weights, torch.Tensor):
            is_weights = is_weights.to(self.device)

        if self.twin:
            q_loss1, q_loss2 = self.get_critic_loss(rollouts)
            q_loss1 = torch.mean(q_loss1 * is_weights)
            q_loss2 = torch.mean(q_loss2 * is_weights)

            self.q_optim1.zero_grad()
            q_loss1.backward()

            self.q_optim2.zero_grad()
            q_loss2.backward()
        else:
            q_loss = self.get_critic_loss(rollouts)
            q_loss = torch.mean(q_loss * is_weights)

            self.q_optim1.zero_grad()
            q_loss.backward()

        policy_loss, pred_log_probs = self.get_actor_loss(rollouts)
        policy_loss = torch.mean(policy_loss * is_weights)

        self.p_optim.zero_grad()
        policy_loss.backward()

        temp_loss = self.get_entropy_loss(pred_log_probs)
        temp_loss = torch.mean(temp_loss * is_weights)

        self.temp_optim.zero_grad()
        temp_loss.backward()

        self.training_steps += 1

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q1 Loss"] = (
                q_loss1.detach().item(), self.training_steps
            )
            self.logger["Train/Policy Loss"] = (
                policy_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Temperature"] = (
                self.temperature, self.training_steps
            )
            self.logger["Train/Action Probability"] = (
                torch.mean(torch.exp(torch.sum(
                    pred_log_probs.detach(), dim=-1
                ))).item(),
                self.training_steps
            )

            # Only log the Q2 if twin
            if(self.twin):
                self.logger["Train/Q2 Loss"] = (
                    q_loss2.detach().item(), self.training_steps
                )

        return self._step_optimizers(rollouts)

    def load(
            self,
            load_path: str = "",
            load_dict: Optional[Dict[str, Any]] = None
        ):
        """
        Loads the algorithm from a given save path. Will use the given state
        dictionary if given, or load from a file otherwise.
        """
        if load_dict is None:
            load_dict = self.load_dict(load_path)

        # Load all the dicts
        super().load(load_path, load_dict)
        self.temperature = torch.exp(self.log_temp).item()
