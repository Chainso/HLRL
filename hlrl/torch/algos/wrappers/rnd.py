import torch

from torch import nn
from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.core.algos import IntrinsicRewardAlgo
from hlrl.torch.algos import TorchRLAlgo
from hlrl.torch.utils.contexts import evaluate

class RND(MethodWrapper, IntrinsicRewardAlgo):
    """
    The Random Network Distillation Algorithm.
    https://arxiv.org/abs/1810.12894
    """
    def __init__(
            self,
            algo: TorchRLAlgo,
            rnd_network: nn.Module,
            rnd_target: nn.Module,
            rnd_optim: Callable[[Tuple[torch.Tensor]], torch.optim.Optimizer],
            normalization_layer: Optional[nn.Module] = None
        ):
        """
        Creates the wrapper to use RND exploration with the algorithm.

        Args:
            algo: The algorithm to wrap.
            rnd_network: The RND network.
            rnd_target: The RND target network
            rnd_optim: The function to create the optimizer for RND.
            normalization_layer: The layer before the RND networks for state
                normalization.
        """
        super().__init__(algo)

        # Done this way to fix pickling issues with torch
        if rnd_network is not None and rnd_target is not None:
            self.rnd = rnd_network
            self.rnd_target = rnd_target
            self.rnd_optim_func = rnd_optim

            self.rnd_loss_func = nn.MSELoss()
            self.normalization_layer = normalization_layer

    def __reduce__(self) -> Tuple[type, Tuple[Any, ...]]:
        """
        Reduces the inputs used to serialize and recreate the RND agent.

        Returns:
            A tuple of the class and input arguments.
        """
        # All tensors should be saved in torch state dict
        return (type(self), (self.obj, None, None, None))

    def create_optimizers(self):
        self.om.create_optimizers()
        self.rnd_optim = self.rnd_optim_func(self.rnd.parameters())

    def _get_loss(self, states):
        """
        Returns the loss of the RND network on the given states.
        """
        states = self.normalization_layer(states)
        rnd_pred = self.rnd(states)

        with torch.no_grad():
            rnd_target = self.rnd_target(states)

        rnd_loss = self.rnd_loss_func(rnd_pred, rnd_target)

        return rnd_loss

    def train_processed_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
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

        return self.om.train_processed_batch(rollouts, *args, **kwargs)

    def intrinsic_reward(
            self,
            state: Any,
            algo_step: OrderedDict[str, Any],
            reward: Any,
            terminal: Any,
            next_state: Any
        ) -> Any:
        """
        Computes the RND loss of the next states

        Args:
            state: The state of the environment.
            algo_step: The transformed algorithm step of the state.
            reward: The reward from the environment.
            terminal: If the next state is a terminal state.
            next_state: The new state of the environment.

        Returns:
            The RND intrinsic reward on the transition.
        """
        with torch.no_grad():
            with evaluate(self.normalization_layer):
                return self._get_loss(next_state).item()
