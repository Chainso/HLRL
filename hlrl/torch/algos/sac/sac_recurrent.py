import torch
import torch.nn as nn

from .sac import SAC
from hlrl.torch.util import polyak_average

class SACRecurrent(SAC):
    """
    Soft-Actor-Critic with a recurrent network.
    """
    def __init__(self, action_space, q_func, policy, discount, polyak,
                 target_update_interval, q_optim, p_optim, temp_optim,
                 twin=True, burn_in_length=0, logger=None):
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
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super().__init__(action_space, q_func, policy, discount, polyak,
                         target_update_interval, q_optim, p_optim, temp_optim,
                         twin=twin, logger=logger)

        self.burn_in_length = burn_in_length

    def forward(self, observation, last_action, hidden_state):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.
            last_action (torch.Tensor): The last action taken.
            hidden_state (torch.Tensor): The hidden state.

        Returns:
            The action, Q-val, and new hidden state if there is one.
        """
        # Only going to update the hidden state using the policy hidden state
        action, log_prob, mean, new_hidden = self.policy.sample(observation,
                                                                last_action,
                                                                hidden_state)

        q_val, _ = self.q_func1(observation, action, last_action,
                                hidden_state)

        return action, q_val, new_hidden

    def step(self, observation, last_action, hidden_state):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation (torch.FloatTensor): A single observation from the
                                             environment.
            last_action (torch.Tensor): The last action taken.
            hidden_state (torch.Tensor): The hidden state.

        Returns:
            The action, Q-value of the action and hidden state if applicable
        """
        action, q_val, new_hidden = self(observation, last_action, hidden_state)

        return (action.detach(), q_val.detach(),
                [tens.detach() for tens in new_hidden])

    def reset_hidden_state(self):
        """
        Resets the hidden state for the network.
        """
        return [tens.to(self.log_temp.device)
                for tens in self.policy.reset_hidden_state()]

    def burn_in_hidden_states(self, rollouts):
        """
        Burns in the hidden state and returns the rest of the input.
        """
        states = rollouts["state"]
        actions = rollouts["action"]
        rewards = rollouts["reward"]
        next_states = rollouts["next_state"]
        terminals = rollouts["terminal"]
        last_actions = rollouts["last_action"]
        hidden_states = rollouts["hidden_state"]

        burn_in_states = states[:, :self.burn_in_length]
        burn_in_last_actions = last_actions[:, :self.burn_in_length]

        burn_in_next_states = states[:, self.burn_in_length:self.burn_in_length + 1]
        burn_in_next_last_act = actions[:, self.burn_in_length:self.burn_in_length + 1]

        _, _, _, new_hiddens = self.policy.sample(burn_in_states,
                                                  burn_in_last_actions,
                                                  hidden_states)
        _, _, _, next_hiddens = self.policy.sample(burn_in_next_states,
                                                   burn_in_next_last_act,
                                                   new_hiddens)

        states = states[:, self.burn_in_length:]
        actions = actions[:, self.burn_in_length:]
        rewards = rewards[:, self.burn_in_length:]
        next_states = next_states[:, self.burn_in_length:]
        terminals = terminals[:, self.burn_in_length:]
        last_actions = last_actions[:, self.burn_in_length:]

        return (states, actions, rewards, next_states, terminals, last_actions,
                new_hiddens, next_hiddens)

    def train_batch(self, rollouts, is_weights):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t, la, h, nh) of training data
                               for the network.
        """
        # Get all the parameters from the rollouts
        (states, actions, rewards, next_states, terminals, last_actions,
         hidden_states, next_hiddens) = self.burn_in_hidden_states(rollouts)

        (full_states, _, full_rewards, full_next_states, full_terminals,
         full_last_actions, full_hidden_states) = rollouts

        q_loss_func = nn.MSELoss()

        with torch.no_grad():
            (next_actions, next_log_probs,
             next_mean, _) = self.policy.sample(next_states, actions,
                                                next_hiddens)
            q_targ_pred1, _ = self.q_func_targ1(next_states, next_actions,
                                                actions, next_hiddens)

        (pred_actions, pred_log_probs,
         pred_means, _) = self.policy.sample(states, last_actions,
                                             hidden_states)

        p_q_pred1, _ = self.q_func1(states, pred_actions, last_actions,
                                    hidden_states)

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if(self._twin):
            with torch.no_grad():
                q_targ_pred2, _ = self.q_func_targ2(next_states, next_actions,
                                                    actions, next_hiddens)

            q_targ = (torch.min(q_targ_pred1, q_targ_pred2)
                      - self._temperature * next_log_probs)

            q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred2, _ = self.q_func2(states, pred_actions, last_actions,
                                        hidden_states)
            p_q_pred = torch.min(p_q_pred1, p_q_pred2)

            q_pred2, _ = self.q_func2(states, actions, last_actions,
                                      hidden_states)
            q_loss2 = q_loss_func(q_pred2, q_next)

            self.q_optim2.zero_grad()
            q_loss2.backward()
            self.q_optim2.step()
        else:
            q_targ = q_targ_pred1 - self._temperature * next_log_probs
            q_next = rewards + (1 - terminals) * self._discount * q_targ

            p_q_pred = p_q_pred1

        q_pred1, _ = self.q_func1(states, actions, last_actions, hidden_states)
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
            self.logger["Train/Q1_Loss"] = (q_loss1, self.training_steps)
            self.logger["Train/Policy_Loss"] = (p_loss, self.training_steps)
            self.logger["Train/Temperature"] = (self._temperature,
                                                self.training_steps)

            # Only log the Q2
            if(self._twin):
                self.logger["Train/Q2_Loss"] = (q_loss2, self.training_steps)

        # Get the new q value to update the experience replay
        with torch.no_grad():
            (updated_actions, new_log_pis, mean,
             new_hidden) = self.policy.sample(full_states, full_last_actions,
                                              full_hidden_states)
                                     
            (u_new_acts, u_new_log_probs, u_new_mean,
             _) = self.policy.sample(full_next_states, updated_actions,
                                     new_hidden)

            new_qs, _ = self.q_func1(full_states, updated_actions,
                                     full_last_actions, full_hidden_states)
            q_targ, _ = self.q_func1(full_next_states, u_new_acts,
                                     updated_actions, new_hidden)

            #new_q_targ = new_qs - self._temperature * new_log_pis
            q_next = (full_rewards + (1 - full_terminals)
                                     * self._discount * q_targ)

        # Update the target
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)
            polyak_average(self.q_func2, self.q_func_targ2, self._polyak)

        self.training_steps += 1
        return new_qs, q_next
