from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from hlrl.core.logger import Logger
from hlrl.torch.common import polyak_average
from hlrl.torch.common.functional import initialize_bias

from .sac import SAC

class SACHybrid(SAC):
    """
    A mixture of the following:
        Discrete SAC https://arxiv.org/pdf/1910.07207.pdf
        Hybrid SAC https://arxiv.org/pdf/1912.11077.pdf
    """
    def __init__(
            self,
            action_parameter_space: Tuple[int, ...],
            discrete_action_space: Tuple[int, ...],
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
        Creates the hybrid soft actor-critic algorithm with the given
        parameters.

        Args:
            action_parameter_space: The action space of the continuous action
                parameters for each discrete action.
            discrete_action_space: The action space of the discrete actions.
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
        super().__init__(
            action_parameter_space, q_func, policy, discount, polyak,
            target_update_interval, q_optim, p_optim, temp_optim, twin, device,
            logger
        )

        # All constants
        self.action_parameter_space = nn.Parameter(
            torch.LongTensor(action_parameter_space), requires_grad=False
        )
        self.num_action_parameters = self.action_parameter_space.sum().item()

        self.action_parameter_offsets = nn.Parameter(
            torch.cat([
                torch.zeros(1), self.action_parameter_space
            ]).cumsum(dim=0),
            requires_grad=False
        )

        self.discrete_action_space = nn.Parameter(
            torch.LongTensor(discrete_action_space), requires_grad=False
        )
        self.num_discrete_actions = self.discrete_action_space.prod().item()

        # Make sure at least the first layer has zero bias so that gradients
        # between actions and parameters of the other actions stay 0
        self.policy.apply(initialize_bias(nn.init.zeros_))

        # Entropy tuning, starting at 1 due to auto-tuning
        """
        self.temperature = 0.2
        self.target_entropy = -torch.sum(torch.tensor(
            action_parameter_space
        )).item()
        self.log_temp = nn.Parameter(
            torch.log(torch.ones(1) * self.temperature), requires_grad=True
        )
        """

        self.discrete_temperature = 1
        self.discrete_target_entropy = 0.98 * torch.log(
            self.discrete_action_space.prod().float()
        ).item()
        self.discrete_log_temp = nn.Parameter(
            torch.log(torch.ones(1) * self.discrete_temperature),
            requires_grad=True
        )

        self.probs = nn.Softmax(-1)

    def create_optimizers(self) -> None:
        """
        Creates the optimizers for the model.
        """
        super().create_optimizers()

        self.discrete_temp_optim = self.temp_optim_func(
            [self.discrete_log_temp]
        )

    def after_update(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recomputes the new losses and updates particular networks parameters
        after a gradient update.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The updated Q-values and target Q-values.
        """
        self.discrete_temperature = torch.exp(self.discrete_log_temp).item()

        return super().after_update(rollouts)

    def get_discrete_entropy(
            self,
            discrete_probs: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the entropy for the discrete actions.

        Args:
            discrete_probs: The probabilities of the discrete actions.

        Returns:
            The discrete portion of the temperature.
        """
        # For numerical stability
        zero_probs_epsilon = (discrete_probs == 0.0).float() * 1e-8
        discrete_log_probs = torch.log(discrete_probs + zero_probs_epsilon)

        entropy = torch.sum(
            discrete_probs * discrete_log_probs, dim=-1, keepdim=True
        )
        entropy = self.discrete_temperature * entropy

        return entropy

    def get_continuous_entropy(
            self,
            cont_log_prob: torch.Tensor,
            discrete_probs: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the entropy of the action parameters.

        Args:
            cont_log_prob: The log probabilities of the action parameter sample.
            discrete_probs: The probabilities of the discrete actions.

        Returns:
            The continuous portion of the entropy.
        """
        # Each discrete action can have a different number of action parameters
        # so can't contain them all in a single tensor
        cont_temp = torch.zeros(
            cont_log_prob.shape[0], 1, device=cont_log_prob.device
        )

        for i in range(self.num_discrete_actions):
            start, end = self.action_parameter_offsets[i:i + 2].long()

            param_log_prob = torch.sum(
                cont_log_prob[..., start:end], dim=-1, keepdim=True
            )

            param_log_prob = discrete_probs[..., i:i + 1] * param_log_prob

            cont_temp += param_log_prob

        cont_temp = self.temperature * cont_temp

        return cont_temp

    def get_entropy(
            self,
            cont_log_prob: torch.Tensor,
            discrete_probs: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the entropy for the continuous log probability of the sample and
        the discrete probabilities.

        Args:
            cont_log_prob: The log probabilities of the action parameter sample.
            discrete_probs: The probabilities of the discrete actions.

        Returns:
            The combined temperature of the discrete and continuous portions.
        """
        discrete_temp = self.get_discrete_entropy(discrete_probs)
        cont_temp = self.get_continuous_entropy(cont_log_prob, discrete_probs)

        total_temp = discrete_temp + cont_temp

        return total_temp

    def get_q_values(
            self,
            q_func: nn.Module,
            states: torch.Tensor,
            action_parameters: torch.Tensor
        ) -> Tuple[torch.Tensor, Distribution]:
        """
        Returns the Q-values for the Q-function on the multi-pass inputs.

        Args:
            q_func: The Q-function to use.
            states: The states to get the Q-values for.
            action_parameters: The action parameters for each discrete action.

        Returns:
            The Q-values and their distribution.
        """
        q_val = q_func(states, action_parameters)

        probs = self.probs(q_val)
        dist = Categorical(probs)

        return q_val, dist

    def get_action(
            self,
            observation: torch.Tensor,
            q_func: nn.Module
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                   torch.Tensor]:
        """
        Passes the observation through the networking using the Q-function
        given.

        Args:
            observation: A batch of observations from the environment.
            q_func: The Q-function to use to get the discrete action.

        Returns:
            The action, Q-value, action parameters, continuous log probability
            of the action parameters, the probability distribution of the
            discrete actions.
        """
        action_parameters, cont_log_prob, _ = self.policy(observation)

        q_val, dist = self.get_q_values(q_func, observation, action_parameters)

        action = dist.sample()
        action = action.unsqueeze(-1)

        return action, q_val, action_parameters, cont_log_prob, dist.probs

    def forward(
            self,
            observation: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the model output for a batch of observations.

        Args:
            observation: A batch of observations from the environment.

        Returns:
            The action, Q-value of the action and the action parameters.
        """
        action, q_val, action_parameters = self.get_action(
            observation, self.q_func1
        )[:3]

        q_val = q_val.gather(-1, action)
        return action, q_val, action_parameters

    def step(
            self,
            observation: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the model action for a single observation of the environment.

        Args:
            observation: A single observation from the environment.

        Returns:
            The action, Q-value of the action and the action parameters.
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
        action_parameters = rollouts["action_parameter"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]

        with torch.no_grad():
            (
                next_actions, q_targ_pred, next_action_params, next_log_probs,
                discrete_probs
            ) = self.get_action(next_states, self.q_func_targ1)

            if self.twin:
                q_targ_pred2, discrete_dist2 = self.get_q_values(
                    self.q_func_targ2, next_states, next_action_params
                )
                q_targ_pred = torch.min(q_targ_pred, q_targ_pred2)

                discrete_probs = torch.min(discrete_probs, discrete_dist2.probs)

            q_entropy = self.get_entropy(next_log_probs, discrete_probs)

            q_targ = q_targ_pred - q_entropy
            q_targ = q_targ.gather(-1, next_actions)

            q_next = rewards + (1 - terminals) * self._discount * q_targ

        q_pred = self.get_q_values(
            self.q_func1, states, action_parameters
        )[0]
        q_pred = q_pred.gather(-1, actions)
        q_loss = self.q_loss_func(q_pred, q_next)

        if self.twin:
            q_pred2 = self.get_q_values(
                self.q_func2, states, action_parameters
            )[0]
            q_pred2 = q_pred2.gather(-1, actions)
            q_loss2 = self.q_loss_func(q_pred2, q_next)

            q_loss = (q_loss, q_loss2)

        return q_loss
        
    def get_actor_loss(
            self,
            rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for the actor/policy.

        Args:
            rollouts: The (s, a, r, s', t) of training data for the network.

        Returns:
            The batch-wise loss for the actor/policy and the log probability of
            the sampled action parameters, and probabilities of the sampled
            action.
        """
        states = rollouts["state"]

        (
            p_q_pred, pred_action_params, pred_log_probs,
            discrete_probs
        ) = self.get_action(states, self.q_func_targ1)[1:]


        if self.twin:
            p_q_pred2, discrete_dist2 = self.get_q_values(
                self.q_func2, states, pred_action_params
            )
            p_q_pred = torch.min(p_q_pred, p_q_pred2)

            discrete_probs = torch.min(discrete_probs, discrete_dist2.probs)

        q_entropy = self.get_entropy(pred_log_probs, discrete_probs)

        p_loss = q_entropy - p_q_pred
        p_loss = p_loss.sum(dim=-1)

        return p_loss, pred_log_probs, discrete_probs

    def get_entropy_loss(
            self,
            pred_log_probs: torch.Tensor,
            discrete_probs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for discrete and continuous entropies.

        Args:
            pred_log_probs: The log probabilities of actions of the current
                policy on the state.
            discrete_probs: The probabilities for the discrete actions.

        Returns:
            The loss for the discrete and continuous entropies.
        """
        # Tune temperature
        with torch.no_grad():
            cont_targ_entropy = self.get_continuous_entropy(
                pred_log_probs, discrete_probs
            )
            cont_targ_entropy /= self.temperature
            cont_targ_entropy += self.target_entropy

            discrete_targ_entropy = self.get_discrete_entropy(discrete_probs)
            discrete_targ_entropy /= self.discrete_temperature
            discrete_targ_entropy += self.discrete_target_entropy

        continuous_temp_loss = -self.log_temp * cont_targ_entropy
        discrete_temp_loss = -self.discrete_log_temp * discrete_targ_entropy

        return continuous_temp_loss, discrete_temp_loss

    def train_processed_batch(
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
        if self.twin:
            q_loss1, q_loss2 = self.get_critic_loss(rollouts)

            q_loss1 = torch.mean(q_loss1 * is_weights)
            q_loss2 = torch.mean(q_loss2 * is_weights)

            q_loss = q_loss1 + q_loss2
        else:
            q_loss = self.get_critic_loss(rollouts)
            q_loss = torch.mean(q_loss * is_weights)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        policy_loss, pred_log_probs, pred_discrete_probs = self.get_actor_loss(
            rollouts
        )
        policy_loss = torch.mean(policy_loss * is_weights)

        self.p_optim.zero_grad()
        policy_loss.backward()
        self.p_optim.step()

        continuous_temp_loss, discrete_temp_loss = self.get_entropy_loss(
            pred_log_probs, pred_discrete_probs
        )
        continuous_temp_loss = torch.mean(continuous_temp_loss * is_weights)
        discrete_temp_loss = torch.mean(discrete_temp_loss * is_weights)

        self.temp_optim.zero_grad()
        continuous_temp_loss.backward()
        self.temp_optim.step()

        self.discrete_temp_optim.zero_grad()
        discrete_temp_loss.backward()
        self.discrete_temp_optim.step()

        self.training_steps += 1

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q1 Loss"] = (
                q_loss1.detach().item(), self.training_steps
            )
            self.logger["Train/Policy Loss"] = (
                policy_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Continuous Temperature Loss"] = (
                continuous_temp_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Discrete Temperature Loss"] = (
                discrete_temp_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Temperature"] = (
                self.temperature, self.training_steps
            )
            self.logger["Train/Discrete Temperature"] = (
                self.discrete_temperature, self.training_steps
            )
            self.logger["Train/Action Probability"] = (
                torch.mean(torch.exp(torch.sum(
                    pred_log_probs.detach(), dim=-1
                ))).item(),
                self.training_steps
            )
            self.logger["Train/Discrete Action Probability"] = (
                torch.mean(torch.max(
                    pred_discrete_probs.detach(), dim=-1
                ).values).item(),
                self.training_steps
            )

            # Only log the Q2 if twin
            if(self.twin):
                self.logger["Train/Q2 Loss"] = (
                    q_loss2.detach().item(), self.training_steps
                )

        return self.after_update(rollouts)

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
        self.discrete_temperature = torch.exp(self.discrete_log_temp).item()
