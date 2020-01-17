import torch
import torch.nn as nn

from hlrl.core.algos import RLAlgo

class TorchRLAlgo(RLAlgo, nn.Module):
    """
    An abstract reinforcement learning algorithm.
    """
    def __init__(self, logger=None):
        """
        Creates the reinforcement learning algorithm.

        Args:
            logger (Logger, optional): The logger to log results while training
                                       and evaluating.
        """
        RLAlgo.__init__(self, logger)
        nn.Module.__init__(self)

    def save(self, save_path):
        model_name = "/model-" + str(self.training_steps) + ".pt"
        torch.save(self.save_dict(), save_path + model_name)


