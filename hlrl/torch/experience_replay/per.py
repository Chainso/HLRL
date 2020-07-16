import torch
import numpy as np

from hlrl.core.experience_replay import PER

class TorchPER(PER):
    """
    A Prioritized Experience Replay implementation using torch tensors
    https://arxiv.org/abs/1511.05952
    """
    def _get_error(self, q_val, q_target):
        """
        Computes the error (absolute difference) between the Q-value and the
        target Q-value
        """
        return torch.abs(q_val - q_target)

    def add(self, experience):
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value.

        Args:
            experience (tuple) : The experience dictionary to add to the buffer
        """
        q_val = experience.pop("q_val")
        target_q_val = experience.pop("target_q_val")

        error = self._get_error(q_val, target_q_val)

        current_index = self.priorities.next_index()
        self.experiences[current_index] = experience

        priority = self._get_priority(error)
        self.priorities.add(priority)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        priorities = self.priorities.get_leaves() / self.priorities.sum()
        indices = np.random.multinomial(len(priorities), priorities, size)

        # Transpose the dictionary values
        batch = self.experiences[indices]
        batch = [dict(zip(batch, field)) for field in zip(*batch.values())]
        batch = np.stack(self.experiences[indices], axis=1)
        batch = [torch.cat([*tens]) for tens in batch]

        probabilities = priorities[indices]

        is_weights = np.power(len(self.priorities) * probabilities,
                              -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return batch, indices, is_weights

    def update_priorities(self, indices, q_vals, q_targets):
        """
        Updates the priority of the experiences at the given indices, using the
        errors given.

        Args:
            q_val ([float]): The Q-values of the actions taken

            discounted_next_qs ([float]): The target Q-values
        """
        errors = self._get_error(q_vals, q_targets)

        for index, error in zip(indices, errors):
            index = index.item()
            error = error.item()

            self.update_priority(index, error)