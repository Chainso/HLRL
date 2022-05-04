from typing import Any, Callable, Dict, OrderedDict, Tuple

import torch
from torch import nn

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.core.algos import UDRLAlgo
from hlrl.torch.algos import TorchRLAlgo

class ReturnHorizonAlgo(MethodWrapper, UDRLAlgo):
    """
    Return horizon distallation to choose horizons based on how 
    """
    def __init__(
            self,
            algo: TorchRLAlgo,
            horizon: nn.Module,
            q_func: nn.Module,
            policy: nn.Module,
            horizon_optim: Callable[[Tuple[torch.Tensor, ...]],
                                    torch.optim.Optimizer]
        ):
        """
        Creates the wrapper to use horizon return distallation to choose
        horizons.

        Args:
            algo: The algorithm to wrap.
            horizon: The horizon distillation network.
            q_func: The Q-function to get values from.
            policy: The policy of the algorithm.
            max_horizon_length: The maximum horizon length to choose.
            horizon_optim: The function to create the optimizer for the horizon
                network.
        """
        super().__init__(algo)

        self.horizon = horizon
        self.q_func = q_func
        self.policy = policy

        self.horizon_optim_func = horizon_optim
        self.horizon_loss_func = nn.MSELoss()

    def create_optimizers(self):
        self.om.create_optimizers()
        self.horizon_optim = self.horizon_optim_func(self.horizon.parameters())

    def _get_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor
        ) -> torch.Tensor:
        """
        Returns the loss of the horizon distillation on the given states.

        Args:
            states: The states to get the loss on.
            actions: The actions to the Q-values for.
            next_states: The next states of the experience.

        Returns:
            The loss of the horizon distillation.
        """
        # Make sure there are at least 2 steps for state and next state
        horizon, _, _ = self.horizon(states)
        horizon[horizon == 1] = 2

        return_pred = self.q_func(states, actions, horizon)

        with torch.no_grad():
            next_horizon = horizon - 1
            next_acts = self.policy(states, next_horizon)
            return_pred_targ = self.q_func(next_states, next_acts, next_horizon)

        horizon_loss = self.horizon_loss_func(return_pred, return_pred_targ)

        return horizon_loss

    def train_processed_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """
        Trains the horizon network before training the batch on the algorithm.
        """
        states = rollouts["state"]
        actions = rollouts["action"]
        next_states = rollouts["next_state"]

        horizon_loss = self._get_loss(states, actions, next_states)

        self.horizon_optim.zero_grad()
        horizon_loss.backward()
        self.horizon_optim.step()

        if self.logger is not None:
            self.logger["Train/Horizon Distillation Loss"] = (
                horizon_loss.detach().item(), self.training_steps
            )

        return self.om.train_processed_batch(rollouts, *args, **kwargs)

    def choose_command(
            self,
            state: torch.Tensor
        ) -> Tuple[float, int]:
        """
        Chooses a command for the agent.

        Args:
            state: The state of the environment.

        Returns:
            A tuple of the target return and time horizon.
        """
        with torch.no_grad():
            horizon, _, _ = self.horizon(state).cpu().item()
            action = self.policy(state, horizon)
            target_return = self.q_func(state, action, horizon)

        return (target_return, horizon)
