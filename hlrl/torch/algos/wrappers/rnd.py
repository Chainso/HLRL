import torch

from torch import nn
from typing import Any, Tuple

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.core.algos import IntrinsicRewardAlgo

class RND(MethodWrapper, IntrinsicRewardAlgo):
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

        self.rnd = rnd_network
        self.rnd_target = rnd_target

        self.rnd_optim_func = rnd_optim
        self.rnd_loss_func = nn.MSELoss()

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the experience
        sequence agent.

        Returns:
            A tuple of the class and input arguments.
        """
        return (
            type(self),
            (
                self.obj, self.rnd, self.rnd_target, self.rnd_optim_func,
                self.rnd_loss_func
            )
        )

    def create_optimizers(self):
        self.om.create_optimizers()
        self.rnd_optim = self.rnd_optim_func(self.rnd.parameters())

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

    def intrinsic_reward(self, state: Any, algo_step: Any, reward: Any,
        next_state: Any):
        """
        Computes the RND loss of the next states

        Args:
            state (Any): The state of the environment.
            action (Any): The last action taken in the environment.
            reward (Any): The external reward to add to.
            next_state (Any): The new state of the environment.
        """
        with torch.no_grad():
            return self._get_loss(next_state).item()
