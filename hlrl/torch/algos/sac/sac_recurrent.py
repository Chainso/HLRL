import torch
import torch.nn as nn

from .sac import SAC
from hlrl.torch.common import polyak_average

class SACRecurrent(SAC):
    """
    Soft-Actor-Critic with a recurrent network.
    """
    def __init__(self, action_space, q_func, policy, discount, polyak,
                 target_update_interval, q_optim, p_optim, temp_optim,
                 twin=True, burn_in_length=0, device="cpu", logger=None):
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
            target_update_interval (int): The number of training steps in
                                          between target updates.
            q_optim (torch.nn.Module) : The optimizer for the Q-function.
            p_optim (torch.nn.Module) : The optimizer for the action policy.
            temp_optim (toch.nn.Module) : The optimizer for the temperature.
            twin (bool, optional) : If the twin Q-function algorithm should be
                                    used, default True.
            burn_in_length (int): The number of samples to "burn in" the hidden
                                  states.
            device (str): The device of the tensors in the module.
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super().__init__(action_space, q_func, policy, discount, polyak,
                         target_update_interval, q_optim, p_optim, temp_optim,
                         twin=twin, device=device, logger=logger)

        self.burn_in_length = burn_in_length

    def forward(self, observation, hidden_state):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.
            hidden_state (torch.Tensor): The hidden state.

        Returns:
            The action, Q-val, and new hidden state if there is one.
        """
        # Only going to update the hidden state using the policy hidden state
        action, log_prob, mean, new_hidden = self.policy(
            observation, hidden_state
        )

        q_val, _ = self.q_func1(observation, action, hidden_state)

        return action, q_val, new_hidden

    def step(self, observation, hidden_state):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation (torch.FloatTensor): A single observation from the
                                             environment.
            hidden_state (torch.Tensor): The hidden state.

        Returns:
            The action, Q-value of the action and hidden state if applicable
        """
        with torch.no_grad():
            action, q_val, new_hidden = self(observation, hidden_state)

        new_hidden = [nh for nh in new_hidden]

        return action, q_val, new_hidden

    def reset_hidden_state(self):
        """
        Resets the hidden state for the network.
        """
        return [
            tens.to(self.log_temp.device)
            for tens in self.policy.reset_hidden_state()
        ]

    def burn_in_hidden_states(self, rollouts):
        """
        Burns in the hidden state and returns the rest of the input.
        """
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        hidden_states = rollouts["hidden_state"]

        burn_in_states = states
        burn_in_next_states = next_states
        new_hiddens = hidden_states

        if self.burn_in_length > 0:    
            with torch.no_grad():
                burn_in_states = states[:, :self.burn_in_length].contiguous()

                _, _, _, new_hiddens = self.policy(
                    burn_in_states, hidden_states
                )

        new_hiddens = [nh for nh in new_hiddens]

        states = states[:, self.burn_in_length:].contiguous()
        actions = actions[:, self.burn_in_length:].contiguous()
        rewards = rewards[:, self.burn_in_length:].contiguous()
        next_states = next_states[:, self.burn_in_length:].contiguous()
        terminals = terminals[:, self.burn_in_length:].contiguous()

        with torch.no_grad():
            first_burned_in = states[:, :1]
            _, _, _, next_hiddens = self.policy(
                first_burned_in, new_hiddens
            )

        next_hiddens = [nh for nh in next_hiddens]

        return (states, actions, rewards, next_states, terminals, new_hiddens,
                next_hiddens)

    def _step_optimizers(self, states, next_states, hidden_states):
        """
        Assumes the gradients have been computed and updates the parameters of
        the network with the optimizers.
        """
        if self.twin:
            self.q_optim2.step()

        self.q_optim1.step()
        self.p_optim.step()
        self.temp_optim.step()

        self._temperature = torch.exp(self.log_temp).item()

        # Get the new q value to update the experience replay
        with torch.no_grad():
            (updated_actions, new_log_pis, _, _) = self.policy(
                states, hidden_states
            )
                                     
            new_qs, _ = self.q_func1(
                states, updated_actions, hidden_states
            )
            new_q_targ = new_qs - self._temperature * new_log_pis

        # Update the target
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)

            if (self.twin):
                polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        return new_qs, new_q_targ

    def train_batch(self, rollouts, is_weights=1):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t, la, h, nh) of training data
                               for the network.
        """
        rollouts = {
            key: value.to(self.device) for key, value in rollouts.items()
        }

        # Switch from (batch size, 2, num layers, hidden size) to
        # (2, num layers, batch size, hidden size)
        rollouts["hidden_state"] = (
            rollouts["hidden_state"].permute(1, 2, 0, 3).contiguous()
        )

        # Get all the parameters from the rollouts
        (states, actions, rewards, next_states, terminals,
         hidden_states, next_hiddens) = self.burn_in_hidden_states(rollouts)

        full_states = rollouts["state"]
        full_next_states = rollouts["next_state"]
        full_hidden_states = rollouts["hidden_state"]

        q_loss_func = nn.MSELoss(reduction='none')

        with torch.no_grad():
            (next_actions, next_log_probs, _, _) = self.policy(
                next_states, next_hiddens
            )

            q_targ_pred1, _ = self.q_func_targ1(
                next_states, next_actions, next_hiddens
            )

        (pred_actions, pred_log_probs,
         _, _) = self.policy(states, hidden_states)

        p_q_pred1, _ = self.q_func1(states, pred_actions, hidden_states)

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if self.twin:
            with torch.no_grad():
                q_targ_pred2, _ = self.q_func_targ2(
                    next_states, next_actions, next_hiddens
                )

                q_targ = (torch.min(q_targ_pred1, q_targ_pred2)
                        - self._temperature * next_log_probs)

                q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred2, _ = self.q_func2(states, pred_actions, hidden_states)
            p_q_pred = torch.min(p_q_pred1, p_q_pred2)

            q_pred2, _ = self.q_func2(states, actions, hidden_states)

            q_loss2 = q_loss_func(q_pred2, q_next) * is_weights
            q_loss2 = torch.mean(q_loss2)

            self.q_optim2.zero_grad()
            q_loss2.backward()
        else:
            with torch.no_grad():
                q_targ = q_targ_pred1 - self._temperature * next_log_probs
                q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred = p_q_pred1

        q_pred1, _ = self.q_func1(states, actions, hidden_states)

        q_loss1 = q_loss_func(q_pred1, q_next) * is_weights
        q_loss1 = torch.mean(q_loss1)

        self.q_optim1.zero_grad()
        q_loss1.backward()

        p_loss = self._temperature * pred_log_probs - p_q_pred
        p_loss = torch.mean(p_loss * is_weights)

        self.p_optim.zero_grad()
        p_loss.backward()

        # Tune temperature
        targ_entropy = pred_log_probs.detach() + self.target_entropy

        temp_loss = self.log_temp * targ_entropy
        temp_loss = -torch.mean(temp_loss * is_weights)

        self.temp_optim.zero_grad()
        temp_loss.backward()

        self._temperature = torch.exp(self.log_temp).item()

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Train/Q1 Loss"] = (
                q_loss1.detach().item(), self.training_steps
            )
            self.logger["Train/Policy Loss"] = (
                p_loss.detach().item(), self.training_steps
            )
            self.logger["Train/Temperature"] = (
                self._temperature, self.training_steps
            )

            # Only log the Q2
            if(self.twin):
                self.logger["Train/Q2 Loss"] = (
                    q_loss2.detach().item(), self.training_steps
            )

        new_qs, new_q_targ = self._step_optimizers(
            full_states, full_next_states, full_hidden_states
        )

        self.training_steps += 1

        return new_qs, new_q_targ
