import torch.nn as nn

from hlrl.core.common.wrappers import MethodWrapper
from hlrl.torch.utils.contexts import evaluating, training

class NormalizeRewardAlgo(MethodWrapper):
    """
    A wrapper to normalize rewards for the algorithm.
    """
    def __init__(
            self,
            algo: TorchRLAlgo
        ):
        """
        Creates the wrapper to use reward normalization exploration for the
        algorithm.

        Args:
            algo: The algorithm to wrap.
        """
        super().__init__(algo)

        self.reward_norm = nn.BatchNorm1d(1)

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

    def process_batch(
            self,
            rollouts: Dict[str, Union[torch.Tensor, np.ndarray]]
        ) -> Dict[str, torch.Tensor]:
        """
        Processes and normalizes the reward for a batch.

        Args:
            rollouts: The training batch with rewards to normalize.

        Returns:
            The processed training batch with normalized rewards.
        """
        rewards = rollouts["reward"]

        # First do the batch norm on training to update running stats, then use
        # eval for the actual batch
        with training(self.reward_norm):
            self.reward_norm(rewards)

        with evaluating(self.reward_norm):
            rollouts["reward"] = self.reward_norm(rewards)
            return self.om.process_batch(rollouts)
