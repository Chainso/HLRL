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
        Multi-Pass Q-Networks https://arxiv.org/pdf/1912.11077.pdf
        Discrete SAC https://arxiv.org/pdf/1910.07207.pdf
        Hybrid SAC https://arxiv.org/pdf/1912.11077.pdf

    Using Hybrid SAC with the multi-pass architecture.
    """
    def __init__(
            self,
            action_parameter_space: Tuple,
            discrete_action_space: Tuple,
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
        self.action_parameter_space = torch.tensor(
            action_parameter_space, device=device
        )
        self.num_action_parameters = self.action_parameter_space.sum().item()

        self.action_parameter_offsets = torch.cat([
            torch.zeros(1, device=self.action_parameter_space.device),
            self.action_parameter_space
        ]).cumsum(dim=0)

        self.discrete_action_space = torch.tensor(
            discrete_action_space, device=device
        )
        self.num_discrete_actions = self.discrete_action_space.prod().item()

        # Make sure at least the first layer has zero bias so that gradients
        # between actions and parameters of the other actions stay 0
        self.policy.apply(initialize_bias(nn.init.zeros_))

        # Entropy tuning, starting at 1 due to auto-tuning
        self.discrete_temperature = 1
        self.discrete_target_entropy = 0.98 * torch.log(
            self.discrete_action_space.prod().float()
        ).item()
        self.discrete_log_temp = nn.Parameter(
            torch.zeros(1), requires_grad=True
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
        states = rollouts["state"]

        self.q_optim1.step()

        if self.twin:
            self.q_optim2.step()

        self.p_optim.step()
        self.temp_optim.step()
        self.discrete_temp_optim.step()

        self.temperature = torch.exp(self.log_temp).item()
        self.discrete_temperature = torch.exp(self.discrete_log_temp).item()

        # Get the new q value to update the experience replay
        with torch.no_grad():
            cont_log_prob, discrete_probs = self.get_action(
                states, self.q_func1
            )[-2:]

            entropy = self.get_entropy(cont_log_prob, discrete_probs)
            zero_target = torch.zeros_like(entropy)

        # Update the target
        if self.training_steps % self._target_update_interval == 0:
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)

            if self.twin:
                polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        return entropy, zero_target

    def make_multipass_input(
            self,
            states: torch.Tensor,
            action_parameters: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the multi-pass input for the Q-networks.

        Args:
            states: The states to create multi-pass inputs on.
            action_parameters: The action parameters to pass to the Q-functions.

        Returns:
            Returns the states and action parameters prepared for the multi-pass
            inputs.
        """
        batch_size = states.shape[0]

        states = states.repeat_interleave(self.num_discrete_actions, dim=0)

        single_scatter_idxs = torch.arange(
            self.num_discrete_actions, device=self.device
        )
        single_scatter_idxs = single_scatter_idxs.repeat_interleave(
            self.action_parameter_space, dim=0
        )
        single_scatter_idxs = single_scatter_idxs.unsqueeze(0).expand(
            batch_size, -1
        )

        scatter_offsets = torch.arange(batch_size, device=self.device)
        scatter_offsets *= self.num_discrete_actions
        scatter_offsets = scatter_offsets.unsqueeze(-1)

        scatter_idxs = scatter_offsets + single_scatter_idxs

        parameter_cat = torch.zeros(
            batch_size * self.num_discrete_actions, self.num_action_parameters,
            device=self.device
        )
        parameter_cat = parameter_cat.scatter(
            0, scatter_idxs, action_parameters
        )

        return states, parameter_cat

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

            ap_param_log_prob = torch.sum(
                cont_log_prob[:, start:end], dim=-1, keepdim=True
            )
            param_log_prob = discrete_probs[:, i:i + 1] * ap_param_log_prob

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
        q_val = q_val.view(
            states.shape[0] // self.num_discrete_actions,
            self.num_discrete_actions
        )

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
        ap_observation, ap_action_parameters = self.make_multipass_input(
            observation, action_parameters
        )

        q_val, dist = self.get_q_values(
            q_func, ap_observation, ap_action_parameters
        )

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
                next_ap_states, next_action_params = self.make_multipass_input(
                    next_states, next_action_params
                )

                q_targ_pred2, discrete_dist2 = self.get_q_values(
                    self.q_func_targ2, next_ap_states, next_action_params
                )
                q_targ_pred = torch.min(q_targ_pred, q_targ_pred2)

                discrete_probs = torch.min(discrete_probs, discrete_dist2.probs)

            q_entropy = self.get_entropy(next_log_probs, discrete_probs)

            q_targ = q_targ_pred - q_entropy
            q_targ = q_targ.gather(-1, next_actions)

            q_next = rewards + (1 - terminals) * self._discount * q_targ


        ap_states, pred_action_parameters = self.make_multipass_input(
            states, action_parameters
        )

        q_pred = self.get_q_values(
            self.q_func1, ap_states, pred_action_parameters
        )[0]
        q_pred = q_pred.gather(-1, actions)
        q_loss = self.q_loss_func(q_pred, q_next)

        if self.twin:
            q_pred2 = self.get_q_values(
                self.q_func2, ap_states, pred_action_parameters
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
            ap_states, pred_action_params = self.make_multipass_input(
                states, pred_action_params
            )

            p_q_pred2, discrete_dist2 = self.get_q_values(
                self.q_func2, ap_states, pred_action_params
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

        policy_loss, pred_log_probs, pred_discrete_probs = self.get_actor_loss(
            rollouts
        )
        policy_loss = torch.mean(policy_loss * is_weights)

        self.p_optim.zero_grad()
        policy_loss.backward()

        continuous_temp_loss, discrete_temp_loss = self.get_entropy_loss(
            pred_log_probs, pred_discrete_probs
        )
        continuous_temp_loss = torch.mean(continuous_temp_loss * is_weights)
        discrete_temp_loss = torch.mean(discrete_temp_loss * is_weights)

        self.temp_optim.zero_grad()
        continuous_temp_loss.backward()

        self.discrete_temp_optim.zero_grad()
        discrete_temp_loss.backward()

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
            self.logger["Train/Discrete Temperature"] = (
                self.discrete_temperature, self.training_steps
            )
            self.logger["Train/Action Probability"] = (
                torch.mean(torch.exp(pred_log_probs.detach())).item(),
                self.training_steps
            )
            self.logger["Train/Discrete Action Probability"] = (
                torch.mean(torch.exp(pred_discrete_probs.detach())).item(),
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
        self.discrete_temperature = torch.exp(self.discrete_log_temp).item()
