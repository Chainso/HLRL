import torch.nn as nn

from copy import deepcopy

from hlrl.core.utils import MethodWrapper
from hlrl.torch.algos import TorchRLAlgo

class RND(MethodWrapper, TorchRLAlgo):
    """
    The Random Network Distillation Algorithm
    https://arxiv.org/abs/1810.12894
    """
    def __init__(self, algo, rnd_network, rnd_optim):
        """
        Creates the wrapper to use RND exploration with the algorithm.

        Args:
            algo (TorchRLAlgo): The algorithm to run.
            rnd_network (torch.nn.Module): The RND network.
            rnd_optim (callable): The function to create the optimizer for RND.
        """
        super().__init__(algo)

        self.rnd = rnd_network

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight.data)

        self.rnd_target = deepcopy(self.rnd).apply(init_weights)
        self.rnd_optim = rnd_optim(self.rnd.parameters())

    def _get_loss(self, states):
        """
        Returns the loss of the RND network on the given states.
        """
        rnd_loss_func = nn.MSELoss()

        rnd_pred = self.rnd(states)
        rnd_target = self.rnd_target(states)

        rnd_loss = rnd_loss_func(rnd_pred, rnd_target)

        return rnd_loss

    def train_batch(self, rollouts, *training_args):
        """
        Trains the RND network before training the batch on the algorithm.
        """
        _, _, _, next_states, _ = rollouts
        rnd_loss = self._get_loss(next_states)

        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        if self.logger is not None:
            self.logger["Train/RND Loss"] = (rnd_loss, self.training_steps)

        return self.om.train_batch(rollouts, *training_args)

    def intrinsic_reward(self, states):
        """
        Computes the intrinsic reward of the states.
        """
        return self._get_loss(states).detach()

    def save_dict(self):
        """
        Adds the rnd network to the save dict of the algorithm.
        """
        state_dict = self.obj.__class__.save_dict(self.obj)
        state_dict["rnd"] = self.rnd.state_dict()
        state_dict["rnd_target"] = self.rnd_target.state_dict()
        state_dict["rnd_optim"] = self.rnd_optim.state_dict()

        return state_dict