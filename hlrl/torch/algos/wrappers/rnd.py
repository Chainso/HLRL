import torch
import torch.nn as nn
from functools import partial
from numpy import sqrt

from hlrl.core.algos import IntrinsicRewardAlgo
from hlrl.torch.common.functional import initialize_weights


class RND(IntrinsicRewardAlgo):
    """
    The Random Network Distillation Algorithm
    https://arxiv.org/abs/1810.12894
    """
    def __init__(self, algo, rnd_network, rnd_target, rnd_optim):
        """
        Creates the wrapper to use RND exploration with the algorithm.

        Args:
            algo (TorchRLAlgo): The algorithm to wrap.
            rnd_network (torch.nn.Module): The RND network.
            rnd_target (torch.nn.Module): The RND target network
            rnd_optim (callable): The function to create the optimizer for RND.
        """
        super().__init__(algo)

        init_fn = initialize_weights(partial(nn.init.orthogonal_, gain=sqrt(2)))

        self.rnd = rnd_network
        self.rnd.apply(init_fn)

        self.rnd_target = rnd_target
        self.rnd_target.apply(init_fn)

        self.rnd_optim = rnd_optim(self.rnd.parameters())
        self.rnd_loss_func = nn.MSELoss()

    def _get_loss(self, states):
        """
        Returns the loss of the RND network on the given states.
        """
        rnd_pred = self.rnd(states)

        with torch.no_grad():
            rnd_target = self.rnd_target(states)

        rnd_loss = self.rnd_loss_func(rnd_pred, rnd_target)

        return rnd_loss

    def train_batch(self, rollouts, *training_args):
        """
        Trains the RND network before training the batch on the algorithm.
        """

        next_states = rollouts["next_state"]

        rnd_loss = self._get_loss(next_states)

        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        if self.logger is not None:
            self.logger["Train/RND Loss"] = (
                rnd_loss.detach().item(), self.training_steps
            )

        return self.om.train_batch(rollouts, *training_args)

    def intrinsic_reward(self, states):
        """
        Computes the intrinsic reward of the states.
        """
        with torch.no_grad():
            return self._get_loss(states).item()

    def save_dict(self):
        """
        Adds the rnd network to the save dict of the algorithm.
        """
        state_dict = self.om.save_dict()
        state_dict["rnd"] = self.rnd.state_dict()
        state_dict["rnd_target"] = self.rnd_target.state_dict()
        state_dict["rnd_optim"] = self.rnd_optim.state_dict()

        return state_dict

    def load(self, load_path, load_dict=None):
        if load_dict is None:
            load_dict = self.load_dict(load_path)

        self.om.load(load_path, load_dict)