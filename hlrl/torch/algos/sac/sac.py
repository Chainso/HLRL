import torch
import torch.nn as nn

from copy import deepcopy

from hlrl.torch.algos import TorchRLAlgo
from hlrl.torch.util import polyak_average


class SAC(TorchRLAlgo):
    """
    The Soft Actor-Critic algorithm from https://arxiv.org/abs/1801.01290
    """

    def __init__(self, q_func, policy, value, discount, ent_coeff,
                 polyak, target_update_interval, q_optim, p_optim, v_optim,
                 twin=True, logger=None):
        """
        Creates the soft actor-critic algorithm with the given parameters

        Args:
            q_func (torch.nn.Module) : The Q-function that takes in the
                                       observation and action.
            policy (torch.nn.Module) : The action policy that takes in the
                                       observation.
            value (torch.nn.Module) : The value network that takes in the
                                      observation.
            discount (float) : The coefficient for the discounted values
                                (0 < x < 1).
            ent_coeff (float) : The coefficient of the entropy reward
                                (0 < x < 1).
            polyak (float) : The coefficient for polyak averaging (0 < x < 1).
            target_update_interval (int): The number of training steps in
                                          between target updates.
            q_optim (torch.nn.Module) : The class of the optimizer for the
                                        Q-function.
            p_optim (torch.nn.Module) : The class of the optimizer for the
                                        action policy.
            v_optim (torch.nn.Module) : The class of the optimizer for the
                                        Q-function.
            twin (bool, optional) : If the twin Q-function algorithm should be
                                    used, default True.
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super().__init__(logger)

        # All constants
        self._discount = discount
        self._ent_coeff = ent_coeff
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

        self.value = value
        self.value_targ = deepcopy(value)
        self.v_optim = v_optim(self.value.parameters())

    def train_from_buffer(self, experience_replay, batch_size, save_path=None,
                          save_interval=10000):
        """
        Starts training the network.

        Args:
            experience_replay (ExperienceReplay): The experience replay buffer
                                                  to sample experiences from.
            num_batches (int): The number of batches to train for.
            batch_size (int): The batch size of the experiences to train on.
            save_path (Optional, str): The path to save the model to.
            save_interval (int): The number of batches between saves.
        """
        if(batch_size <= len(experience_replay)):
            sample = experience_replay.sample(batch_size)
            rollouts, idxs, is_weights = sample

            new_q, new_q_targ = self.train_batch(rollouts, is_weights)
            experience_replay.update_priorities(idxs, new_q, new_q_targ)

            if(save_path is not None
               and self.training_steps % save_interval == 0):
                self.save(save_path)

    def forward(self, observation):
        """
        Get the model output for a batch of observations

        Args:
            observation (torch.FloatTensor): A batch of observations from the
                                             environment.

        Returns:
            The action, Q-value and state value.
        """
        action = self.policy(observation)
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
        action, mean, log_probs = self.policy.sample(observation)
        q_val = self.q_func1(observation, action)

        return action.detach(), q_val.detach()

    def train_batch(self, rollouts, is_weights):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t, idx, is_weights) of
                               training data for the network.
        """
        # Get all the parameters from the rollouts
        states, actions, rewards, next_states, terminals = rollouts

        q_loss_func = nn.MSELoss()
        v_loss_func = nn.MSELoss()

        value_targ_next_pred = (1 - terminals) * self.value_targ(next_states)
        value_targ_next_pred = value_targ_next_pred.detach()
        new_actions, mean, log_pis = self.policy.sample(states)

        entropy = self._ent_coeff * log_pis

        q_targ_pred1 = self.q_func_targ1(states, new_actions).detach()
        q_targ = rewards + self._discount * value_targ_next_pred
        q_loss1 = q_loss_func(self.q_func1(states, actions), q_targ)

        self.q_optim1.zero_grad()
        q_loss1.backward()
        self.q_optim1.step()

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if(self._twin):
            q_targ_pred2 = self.q_func_targ2(states, new_actions).detach()
            q_loss2 = q_loss_func(self.q_func2(states, actions), q_targ)

            self.q_optim2.zero_grad()
            q_loss2.backward()
            self.q_optim2.step()

            value_targ = torch.min(
                q_targ_pred1, q_targ_pred2) - entropy.detach()
        else:
            value_targ = q_targ_pred1 - entropy.detach()

        v_loss = v_loss_func(self.value(states), value_targ)

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        p_loss = torch.mean(entropy - q_targ_pred1)

        self.p_optim.zero_grad()
        p_loss.backward()
        self.p_optim.step()

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Q1 Loss"] = q_loss1, self.training_steps
            self.logger["Policy Loss"] = p_loss, self.training_steps
            self.logger["Value Loss"] = v_loss, self.training_steps

        # Only log the Q2
        if(self._twin and self.logger is not None):
            self.logger["Q2 Loss"] = q_loss2, self.training_steps

        # Get the new q value to update the experience replay
        updated_actions, mean, new_log_pi = self.policy.sample(states)
        new_qs = self.q_func1(states, updated_actions).detach()
        new_value_targ = (1 - terminals) * \
            self.value_targ(next_states).detach()
        new_q_targ = rewards + self._discount * value_targ_next_pred

        # Update the target
        if (self.training_steps % self._target_update_interval == 0):
            polyak_average(self.q_func1, self.q_func_targ1, self._polyak)
            polyak_average(self.q_func2, self.q_func_targ2, self._polyak)
            polyak_average(self.value, self.value_targ, self._polyak)

        self.training_steps += 1
        return new_qs, new_q_targ

    def save(self, save_path):
        # Save all the dicts
        state = {
            "env_episodes": self.env_episodes,
            "training_steps": self.training_steps,
            "env_steps": self.env_steps,
            "q_func1": self.q_func1.state_dict(),
            "q_func_targ1": self.q_func_targ1.state_dict(),
            "q_optim1": self.q_optim1.state_dict(),
            "policy": self.policy.state_dict(),
            "p_optim": self.p_optim.state_dict(),
            "value": self.value.state_dict(),
            "value_targ": self.value_targ.state_dict(),
            "v_optim": self.v_optim.state_dict()
        }

        # Save second q function if this is twin sac
        if (self._twin):
            state["q_func2"] = self.q_func2.state_dict()
            state["q_func_targ2"] = self.q_func_targ2.state_dict()
            state["q_optim2"] = self.q_optim2.state_dict()

        torch.save(state, save_path)

    def load(self, load_path):
        state = torch.load(load_path)

        # Load all the dicts
        self.env_episodes = state["env_episodes"]
        self.training_steps = state["training_steps"]
        self.env_steps = state["env_steps"]
        self.q_func1.load_state_dict(state["q_func1"])
        self.q_func_targ1.load_state_dict(state["q_func_targ1"])
        self.q_optim1.load_state_dict(state["q_optim1"])
        self.policy.load_state_dict(state["policy"])
        self.p_optim.load_state_dict(state["p_optim"])
        self.value.load_state_dict(state["value"])
        self.value_targ.load_state_dict(state["value_targ"])
        self.v_optim.load_state_dict(state["v_optim"])

        # Load second q function if this is twin sac
        if (self._twin):
            self.q_func2.load_state_dict(state["q_func2"])
            self.q_func_targ2.load_state_dict(state["q_func_targ2"])
            self.q_optim2.load_state_dict(state["q_optim2"])
