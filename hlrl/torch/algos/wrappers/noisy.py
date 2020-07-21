from hlrl.core.utils import MethodWrapper
from hlrl.torch.layers import NoisyLayer

class NoisyAlgo(MethodWrapper):
    """
    A wrapper to reset the noise of a noisy network after each training batch
    and on reset.
    """
    def __init__(self, algo):
        """
        Transforms the agent into one that uses noise for exploration.

        Args:
            algo (torch.nn.Module): The algorithm to wrap.
        """
        super().__init__(algo)

    def train_batch(self, *training_args):
        """
        Resets the noise after training.
        """
        training_ret = self.om.train_batch(*training_args)

        for module in self.modules():
            if isinstance(module, NoisyLayer):
                module.reset_noise()

        return training_ret