import torch
import torch.nn as nn

from copy import deepcopy

from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.common import polyak_average
from hlrl.torch.common.functional import initialize_weights

class SAC(TorchOffPolicyAlgo):
    """
    The Soft Actor-Critic algorithm from https://arxiv.org/abs/1801.01290
    """
    def __init__(self, action_space, q_func, policy, discount, polyak,
                 target_update_interval, q_optim, p_optim, temp_optim,
                 twin=True, device="cpu", logger=None):
        """
        Creates the soft actor-critic algorithm with the given parameters

        Args:
            action_space (tuple) : The dimensions of the action space of the
                                   environment.
            q_func (torch.nn.Module) : The Q-function that takes in the
                                       observation and action.
            policy (torch.nn.Module) : The action policy that takes in the
                                       observation.
            discount (float) : The coefficient for the discounted values
                                (0 < x < 1).
            polyak (float) : The coefficient for polyak averaging (0 < x < 1).
            target_update_interval (int): The number of environment steps in
                                          between target updates.
            q_optim (torch.nn.Module) : The optimizer for the Q-function.
            p_optim (torch.nn.Module) : The optimizer for the action policy.
            temp_optim (toch.nn.Module) : The optimizer for the temperature.
            twin (bool, optional) : If the twin Q-function algorithm should be
                                    used, default True.
            device (str): The device of the tensors in the module.
            logger (Logger, optional) : The logger to log results while training
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
        self.q_func_targ1 = deepcopy(self.q_func1)

        # Instantiate a second Q-function for twin SAC
        if(self.twin):
            self.q_func2 = deepcopy(q_func).apply(
                initialize_weights(nn.init.xavier_uniform_)
            )
            self.q_func_targ2 = deepcopy(self.q_func2)

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

    def _step_optimizers(self, states):
        """
        Assumes the gradients have been computed and updates the parameters of
        the network with the optimizers.

        Args:
            states: The states to recompute losses on.
        """
        if self.twin:
            self.q_optim2.step()

        self.q_optim1.step()
        self.p_optim.step()
        self.temp_optim.step()

        self.temperature = torch.exp(self.log_temp).item()

        # Get the new q value to update the experience replay
        with torch.no_grad():
            updated_actions, new_log_pis, _ = self.policy(states)
            new_qs = self.q_func1(states, updated_actions)
            new_q_targ = new_qs - self.temperature * new_log_pis

        # Update the target
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)

            if (self.twin):
                polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        
        return new_qs, new_q_targ

    def forward(self, observation):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.

        Returns:
            The action and Q-value.
        """
        action, log_prob, mean = self.policy(observation)
        q_val = self.q_func1(observation, action)

        return action, q_val

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
            return self(observation)

    def train_batch(self, rollouts, is_weights=1):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t) of training data for the
                               network.
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

        q_loss_func = nn.MSELoss(reduction='none')

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy(next_states)
            q_targ_pred1 = self.q_func_targ1(next_states, next_actions)

        pred_actions, pred_log_probs, _ = self.policy(states)

        p_q_pred1 = self.q_func1(states, pred_actions)

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if(self.twin):
            with torch.no_grad():
                q_targ_pred2 = self.q_func_targ2(next_states, next_actions)

                q_targ = (torch.min(q_targ_pred1, q_targ_pred2)
                        - self.temperature * next_log_probs)

                q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred2 = self.q_func2(states, pred_actions)
            p_q_pred = torch.min(p_q_pred1, p_q_pred2)

            q_pred2 = self.q_func2(states, actions)
            q_loss2 = torch.mean(q_loss_func(q_pred2, q_next) * is_weights)

            self.q_optim2.zero_grad()
            q_loss2.backward()
        else:
            with torch.no_grad():
                q_targ = q_targ_pred1 - self.temperature * next_log_probs
                q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred = p_q_pred1

        q_pred1 = self.q_func1(states, actions)
        q_loss1 = torch.mean(q_loss_func(q_pred1, q_next) * is_weights)

        self.q_optim1.zero_grad()
        q_loss1.backward()

        p_loss = self.temperature * pred_log_probs - p_q_pred
        p_loss = torch.mean(p_loss * is_weights)

        self.p_optim.zero_grad()
        p_loss.backward()

        # Tune temperature
        targ_entropy = pred_log_probs.detach() + self.target_entropy
        temp_loss = -torch.mean(self.log_temp * targ_entropy)

        self.temp_optim.zero_grad()
        temp_loss.backward()

        self.training_steps += 1

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q1 Loss"] = (
                q_loss1.detach().item(), self.training_steps
            )
            self.logger["Train/Policy Loss"] = (
                p_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Temperature"] = (
                self.temperature, self.training_steps
            )
            self.logger["Train/Action Probability"] = (
                torch.mean(torch.exp(pred_log_probs.detach())).item(),
                self.training_steps
            )

            # Only log the Q2 if twin
            if(self.twin):
                self.logger["Train/Q2 Loss"] = (
                    q_loss2.detach().item(), self.training_steps
                )

        new_qs, new_q_targ = self._step_optimizers(states)

        return new_qs, new_q_targ

    def save_dict(self):
        # Save all the dicts
        state_dict = super().save_dict()
        state_dict["temperature"] = self.temperature

        return state_dict

    def load(self, load_path, load_dict=None):
        if load_dict is None:
            load_dict = self.load_dict(load_path)

        # Load all the dicts
        super().load(load_path, load_dict)
        self.temperature = load_dict["temperature"]
