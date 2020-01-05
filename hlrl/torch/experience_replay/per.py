import torch
import numpy as np

from hlrl.core.experience_replay import PER
from .binary_sum_tree import TorchBinarySumTree

class TorchPER(PER):
    """
    A Prioritized Experience Replay implementation using torch tensors
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity: int, alpha: float, beta: float,
                 beta_increment: float, epsilon: float, device: str):
        """
        Creates a new PER buffer with the given parameters

        capacity : The capacity of the replay buffer
        alpha : The alpha value for the prioritization, between 0 and 1
                inclusive
        beta : The beta value for the importance sampling, between 0 and 1
               inclusive
        beta_increment : The value to increment the beta by
        epsilon : The value of epsilon to add to the priority
        device (str): The device of the tensors in the replay buffer
        """
        super().__init__(capacity, alpha, beta, beta_increment, epsilon)
        self.device = torch.device(device)

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experiences = np.zeros(capacity, dtype=object)
        self.priorities = TorchBinarySumTree(capacity, device)

    def _get_error(self, q_val, q_target):
        """
        Computes the error (absolute difference) between the Q-value and the
        target Q-value
        """
        return torch.abs(q_val - q_target).item()

    def add(self, experience, q_val, q_target):
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value.

        Args:
            experience (tuple) : The (s, a, r, ...) experience to add to the
                                 buffer

            q_val (float): The Q-value of the action taken

            q_target (float): The target Q-value
        """
        error = self._get_error(q_val, q_target)

        current_index = self.priorities.next_index()
        self.experiences[current_index] = np.array(experience, dtype=object)

        priority = self._get_priority(error)
        self.priorities.add(priority)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        priorities = self.priorities.get_leaves() / self.priorities.sum()
        indices = torch.multinomial(priorities, size)

        batch = np.array(self.experiences[indices], dtype=object)

        probabilities = priorities[indices]

        is_weights = torch.pow(len(self.priorities) * probabilities,
                               -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return batch, indices, is_weights