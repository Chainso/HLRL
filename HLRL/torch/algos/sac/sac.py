import torch
import torch.nn as nn

from copy import deepcopy

from HLRL.torch.algos.base import TorchRLAlgo

class SAC(TorchRLAlgo):
    """
    The Soft Actor-Critic algorithm from https://arxiv.org/abs/1801.01290
    """
    def __init__(self, device, q_func, policy, value, discount, ent_coeff,
                 polyak, q_optim, q_optim_args, p_optim, p_optim_args, v_optim,
                 v_optim_args, twin=True, logger=None):
        """
        Creates the soft actor-critic algorithm with the given parameters

        Args:
            device (str): The device for the algorithm to run on. "cpu" for
                          cpu, and "cuda" for gpu.
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
            q_optim (torch.nn.Module) : The class of the optimizer for the
                                        Q-function.
            q_optim_args (dict) : The dictionary of arguments for the optimizer
                                  of the Q-function except for the module
                                  parameters.
            p_optim (torch.nn.Module) : The class of the optimizer for the
                                        action policy.
            p_optim_args (dict) : The dictionary of arguments for the optimizer
                                  of the policy except for the module
                                  parameters.
            v_optim (torch.nn.Module) : The class of the optimizer for the
                                        Q-function.
            v_optim_args (dict) : The dictionary of arguments for the optimizer
                                  of the value function except for the module
                                  parameters.
            twin (bool, optional) : If the twin Q-function algorithm should be
                                    used, default True.
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating, default None.
        """
        super(self).__init__(device, logger)

        # All constants
        self._discount = discount
        self._ent_coeff = ent_coeff
        self._polyak = polyak
        self._twin = twin

        # The networks
        self.q_func1 = q_func
        self.q_func_targ1 = deepcopy(self.q_func1)
        self.q_optim1 = q_optim(self.q_func1.parameters(), **q_optim_args)

        # Instantiate a second Q-function for twin SAC
        if(self._twin):
            # Re-initialize the weights
            re_init = lambda m: nn.init.xavier_uniform(m.weight.data)

            self.q_func2 = deepcopy(q_func).apply(re_init)
            self.q_func_targ2 = deepcopy(self.q_func2)
            self.q_optim2 = q_optim(self.q_func2.parameters(), **q_optim_args)

        self.policy = policy
        self.p_optim = p_optim(self.policy.parameters(), **p_optim_args)

        self.value = value
        self.value_targ = deepcopy(value)
        self.v_optim = v_optim(self.value.parameters(), **v_optim_args)

    def start_training(self, experience_replay, batch_size):
        """
        Starts training the network.

        Args:
            experience_replay (ExperienceReplay): The experience replay buffer
                                                  to sample experiences from.

            batch_size (int): The batch size of the experiences to train on
        """
        while(self.trainsing):
            sample = experience_replay.sample(batch_size)
            s, a, r, n_s, t, idxs = self.train_batch(sample)

            new_q, new_q_targ = self.train_batch((s, a, r, n_s, t))
            experience_replay.update_priorities(idxs, new_q, new_q_targ)

    def step(self, observation):
        """
        Get the model action for a single observation of gameplay.

        Args:
            observation (torch.FloatTensor): A single observation from the
                                             environment.

        Returns:
            The action from for the observation given.
        """
        action = self.policy(observation, False)
        q_val = self.q_func1(observation, action)

        return action.item(), q_val.item()

    def train_batch(self, rollouts):
        """
        Trains the network for a batch of (state, action, reward, next_state,
        terminals) rollouts.

        Args:
            rollouts (tuple) : The (s, a, r, s', t, idx, is_weights) of training data for the
                               network.
        """
        # Get all the parameters from the rollouts
        rollouts = [torch.from_numpy(roll).to(self.device) for roll in rollouts]
        states, actions, rewards, next_states, terminals = rollouts

        q_loss_func = nn.MSELoss()
        v_loss_func = nn.MSELoss()

        value_targ_next_pred = (1 - terminals) * self.value_targ(next_states)

        new_actions, log_pis = self.policy(states, True)
        entropy = -self.ent_coeff * log_pis

        q_targ_pred1 = self.q_func1(states, new_actions)

        q_targ = rewards + self._discount * value_targ_next_pred
        q_loss1 = q_loss_func(self.q_func1(states, actions), q_targ)

        self.q_optim1.zero_grad()
        q_loss1.backward()
        self.q_optim1.step()

        # Only get the loss for q_func2 if using the twin Q-function algorithm
        if(self._twin):
            q_targ_pred2 = self.q_func_targ2(states, new_actions)     
            q_loss2 = q_loss_func(self.q_func2(states, actions), q_targ)

            self.q_optim2.zero_grad()
            q_loss2.backward()
            self.q_optim2.step()

            value_targ = torch.min(q_targ_pred1, q_targ_pred2) - entropy
        else:
            value_targ = q_targ_pred1 - entropy

        v_loss = v_loss_func(self.value(states), value_targ)

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        p_loss = torch.mean(q_targ_pred1 - entropy)

        self.p_optim.zero_grad()
        p_loss.backward()
        self.p_optim.step()

        # Log the losses if a logger was given
        if(self.logger is not None):
            self.logger["Q1 Loss"] =  q_loss1, self.training_steps
            self.logger["Policy Loss"] = p_loss, self.training_steps
            self.logger["Value Loss"] = v_loss, self.training_steps

        # Only log the Q2 
        if(self._twin and and self.logger is not None):  
            self.logger["Q2 Loss"] = q_loss2, self.training_steps

        new_qs = self.online(states)
        new_value_targ = (1 - terminals) * self.value_targ(next_states)
        new_q_targ = rewards + self._discount * value_targ_next_pred

        return new_qs, new_q_targ
