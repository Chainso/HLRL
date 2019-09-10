import torch

from abc import ABC

from HLRL.core.algos.base import RLAlgo

class TorchRLAlgo(ABC, RLAlgo):
    """
    An abstract reinforcement learning algorithm
    """
    def __init__(self, device, logger=None):
        """
        Args:
            device (str): The device for the algorithm to run on. "cpu" for
                          cpu, and "cuda" for gpu
            logger (Logger, optional) : The logger to log results while training
                                        and evaluating
        """
        super(RLAlgo, self).__init__(logger);

        self._device = torch.device(device)

    @property
    def device(self):
        """
        Returns: 
            The device for the algorithm to run on. "cpu" for cpu, and "cuda"
            for gpu
        """
        return self._device
