from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.algos import TorchRLAlgo
from hlrl.torch.utils.contexts import evaluate, training

class NormalizeReturnAlgo(MethodWrapper):
    """
    A wrapper to normalize rewards for the algorithm.
    """
    def __init__(self, algo: TorchRLAlgo):
        """
        Creates the wrapper to use reward normalization exploration for the
        algorithm.

        Args:
            algo: The algorithm to wrap.
        """
        super().__init__(algo)

        self.reward_norm = nn.BatchNorm1d(
            1, affine=False, track_running_stats=True
        )

    def process_batch(
            self,
            rollouts: Dict[str, torch.Tensor],
            *args: Any,
            **kwargs: Any
        ) -> Dict[str, torch.Tensor]:
        """
        Processes and normalizes the reward for a batch.

        Args:
            rollouts: The training batch with rewards to normalize.
            args: Any positional arguments for the wrapped algorithm to process
                the batch.
            kwargs: Any keyword arguments for the wrapped algorithm to process
                the batch.

        Returns:
            The processed training batch with normalized rewards.
        """
        rollouts, *rest = self.om.process_batch(rollouts, *args, **kwargs)
        rewards = rollouts["reward"]

        # First do the batch norm on training to update running stats, then use
        # eval for the actual batch
        self.reward_norm(rewards)

        with evaluate(self.reward_norm):
            rollouts["reward"] = self.reward_norm(rewards)

        return rollouts, *rest

    def train_processed_batch(
            self,
            rollouts: Dict[str, Any],
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """
        Adds logs for the reward normalization to the batch training.

        Args:
            rollouts: The training batch to process.
            args: Positional training arguments.
            kwargs: Keyword training arguments.

        Returns:
            The underlying algorithm's training procedure.
        """
        train_ret = self.om.train_processed_batch(rollouts, *args, **kwargs)

        # Log running mean and std
        if self.logger is not None:
            self.logger["Train/Return Mean"] = (
                self.reward_norm.running_mean, self.training_steps
            )
            self.logger["Train/Return Std"] = (
                torch.sqrt(self.reward_norm.running_var), self.training_steps
            )

        return train_ret
