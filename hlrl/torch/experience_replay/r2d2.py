import torch
import numpy as np

from .per import TorchPER

class TorchR2D2(TorchPER):
    """
    An implementation of Recurrent Experience Replay using torch.
    https://openreview.net/pdf?id=r1lyTjAqYX
    """
    def __init__(self, capacity: int, alpha: float, beta: float,
                 beta_increment: float, epsilon: float, max_factor: float):
        """
        Args:
            capacity (int): The capacity of the replay buffer.
            alpha (float): The alpha value for the prioritization,
                           between 0 and 1 inclusive.
            beta (float): The beta value for the importance sampling,
                          between 0 and 1 inclusive.
            beta_increment (float): The value to increment the beta by.
            epsilon (float): The value of epsilon to add to the priority.
            max_factor (float): The factor of max error to mean error.
        """
        super().__init__(capacity, alpha, beta, beta_increment, epsilon)

        self.max_factor = max_factor

    def get_error(self, q_val, q_target):
        """
        Computes the error (absolute difference) between the Q-value and the
        target Q-value
        """
        reg_error = torch.abs(q_val - q_target)

        error = (self.max_factor * reg_error.max(dim=1).values
                 + (1 - self.max_factor) * reg_error.mean(dim=1))

        return error