import numpy as np
from typing import Tuple

from .binary_sum_tree import BinarySumTree
from .replay import ExperienceReplay

class PER(ExperienceReplay):
    """
    A Prioritized Experience Replay implementation.
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity: int, alpha: float, beta: float,
                 beta_increment: float, epsilon: float):
        """
        Creates a new PER buffer with the given parameters.

        Args:
            capacity (int): The capacity of the replay buffer.
            alpha (float): The alpha value for the prioritization,
                           between 0 and 1 inclusive.
            beta (float): The beta value for the importance sampling,
                          between 0 and 1 inclusive.
            beta_increment (float): The value to increment the beta by.
            epsilon (float): The value of epsilon to add to the priority.
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experiences = {}
        self.priorities = BinarySumTree(capacity)

    def __len__(self):
        """
        Returns the number of experiences added to the buffer.
        """
        return len(self.priorities)

    def get_priority(self, error):
        """
        Computes the priority for the given error.

        error : The error to get the priority for.
        """
        return (error + self.epsilon) ** self.alpha

    def get_error(self, q_val, q_target):
        """
        Computes the error (absolute difference) between the Q-value plus the
        reward and the discounted Q-value of the next state.
        """
        return np.abs(q_val - q_target)

    def add(self, experience):
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value.

        Args:
            experience (Dict) : The experience dictionary to add to the buffer.
        """
        q_val = experience.pop("q_val")
        target_q_val = experience.pop("target_q_val")

        error = self.get_error(q_val, target_q_val).item()

        current_index = self.priorities.next_index()
    
        # Store individually for faster "zipping"
        for key in experience:
            if key not in self.experiences:
                self.experiences[key] = np.zeros(self.capacity, dtype=object)

            self.experiences[key][current_index] = experience[key]

        priority = self.get_priority(error)

        self.priorities.add(priority)
 
    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        assert(size > 0)

        priorities = self.priorities.get_leaves() / self.priorities.sum()

        # A hack right now
        priorities /= priorities.sum()

        indices = np.random.choice(len(priorities), size, p = priorities)

        batch = {key: np.concatenate(self.experiences[key][indices].tolist())
            for key in self.experiences}

        probabilities = priorities[indices]

        is_weights = np.power(len(self.priorities) * probabilities,
                              -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return batch, indices, is_weights

    def update_priority(self, index: int, error: float):
        """
        Updates the priority of the experience at the given index, using the
        error given.

        Args:
            index: The index of the experience
            error: The new error of the experience
        """
        if index < self.capacity:
            priority = self.get_priority(error)
            self.priorities.set(priority, index)

    def update_priorities(self,
                          indices: Tuple[int, Tuple[int, ...]],
                          q_vals: Tuple[Any, ...],
                          q_targets: Tuple[Any, ...]) -> None:
        """
        Updates the priority of the experiences at the given indices, using the
        errors given.

        Args:
            indices: The indices of the experiences to update.
            q_vals: The updated Q-values of the actions taken.
            q_targets: The new targets for the Q-values.
        """
        errors = self.get_error(q_vals, q_targets)

        for index, error in zip(indices, errors):
            index = index
            error = error

            self.update_priority(index, error)
