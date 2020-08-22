import torch
import torch.nn as nn

from copy import deepcopy

from hlrl.torch.algos import TorchOffPolicyAlgo
from hlrl.torch.utils import polyak_average

class SAC(TorchOffPolicyAlgo):
    """
    The Soft Actor-Critic algorithm from https://arxiv.org/abs/1801.01290
    """
    def __init__(self, action_space, q_func, policy, discount, polyak,
                 target_update_interval, q_optim, p_optim, temp_optim,
                 twin=True, logger=None):
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
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super().__init__(logger)

        # All constants
        self._discount = discount
        self._polyak = polyak
        self._target_update_interval = target_update_interval
        self._twin = twin

        # The networks
        self.q_func1 = q_func
        self.q_func_targ1 = deepcopy(self.q_func1)
        self.q_optim1 = q_optim(self.q_func1.parameters())

        # Instantiate a second Q-function for twin SAC
        if(self._twin):
            def init_weights(m):
                if hasattr(m, "weight"):
                    nn.init.xavier_uniform_(m.weight.data)

            self.q_func2 = deepcopy(q_func).apply(init_weights)
            self.q_func_targ2 = deepcopy(self.q_func2)
            self.q_optim2 = q_optim(self.q_func2.parameters())

        self.policy = policy
        self.p_optim = p_optim(self.policy.parameters())

        # Entropy tuning, starting at 1 due to auto-tuning
        self._temperature = 1
        self.target_entropy = -torch.prod(torch.Tensor(action_space)).item()
        self.log_temp = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.temp_optim = temp_optim([self.log_temp])

    def forward(self, observation):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.

        Returns:
            The action and Q-value.
        """
        action, log_prob, mean = self.policy.sample(observation)
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
        action, q_val = self(observation)

        return action.detach(), q_val.detach()

    def train_batch(self, rollouts, is_weights=None):
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

        q_loss_func = nn.MSELoss()

        with torch.no_grad():
            next_actions, next_log_probs, next_mean = self.policy.sample(next_states)
            q_targ_pred1 = self.q_func_targ1(next_states, next_actions)

        pred_actions, pred_log_probs, pred_means = self.policy.sample(states)

        p_q_pred1 = self.q_func1(states, pred_actions)

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if(self._twin):
            with torch.no_grad():
                q_targ_pred2 = self.q_func_targ2(next_states, next_actions)

            q_targ = (torch.min(q_targ_pred1, q_targ_pred2)
                      - self._temperature * next_log_probs)
            q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred2 = self.q_func2(states, pred_actions)
            p_q_pred = torch.min(p_q_pred1, p_q_pred2)

            q_pred2 = self.q_func2(states, actions)
            q_loss2 = q_loss_func(q_pred2, q_next)

            self.q_optim2.zero_grad()
            q_loss2.backward()
            self.q_optim2.step()
        else:
            q_targ = q_targ_pred1 - self._temperature * next_log_probs
            q_next = rewards + (1 - terminals) * self._discount * q_targ
            p_q_pred = p_q_pred1

        q_pred1 = self.q_func1(states, actions)
        q_loss1 = q_loss_func(q_pred1, q_next)

        self.q_optim1.zero_grad()
        q_loss1.backward()
        self.q_optim1.step()

        p_loss = torch.mean(self._temperature * pred_log_probs - p_q_pred)

        self.p_optim.zero_grad()
        p_loss.backward()
        self.p_optim.step()

        # Tune temperature
        targ_entropy = pred_log_probs.detach() + self.target_entropy
        temp_loss = -torch.mean(self.log_temp * targ_entropy)

        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self._temperature = torch.exp(self.log_temp).item()

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q1 Loss"] = (q_loss1, self.training_steps)
            self.logger["Train/Policy Loss"] = (p_loss, self.training_steps)
            self.logger["Train/Temperature"] = (self._temperature,
                                                self.training_steps)

            # Only log the Q2
            if(self._twin):
                self.logger["Train/Q2 Loss"] = (q_loss2, self.training_steps)

        # Get the new q value to update the experience replay
        with torch.no_grad():
            updated_actions, new_log_pis, mean = self.policy.sample(states)
            new_qs = self.q_func1(states, updated_actions)
            new_q_targ = new_qs - self._temperature * new_log_pis

        # Update the target
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)

            if (self._twin):
                polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        self.training_steps += 1
        return new_qs, new_q_targ

    def save_dict(self):
        # Save all the dicts
        state_dict = super().save_dict()
        state_dict["temperature"] = self._temperature

        return state_dict

    def load(self, load_path, load_dict=None):
        if load_dict is None:
            load_dict = self.load_dict(load_path)

        # Load all the dicts
        super().load(load_path, load_dict)
        self._temperature = load_dict["temperature"]
