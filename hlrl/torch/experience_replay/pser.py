import numpy as np

from .per import TorchPER

class TorchPSER(TorchPER):
    """
    An implementation of Prioritized Sequence Experience Replay using torch.
    https://arxiv.org/pdf/1905.12726.pdf
    """
    def __init__(self, capacity: int, alpha: float, beta: float,
                 beta_increment: float, epsilon: float, threshold=0,
                 decay=0.65, p_scale = 0.7):
        """
        Creates a new PSER buffer with the given parameters. The window size is
        computed as floor(ln(threshold)/ln(decay))

        Args:
            capacity (int): The capacity of the replay buffer.
            alpha (float): The alpha value for the prioritization,
                           between 0 and 1 inclusive.
            beta (float): The beta value for the importance sampling,
                          between 0 and 1 inclusive.
            beta_increment (float): The value to increment the beta by.
            epsilon (float): The value of epsilon to add to the priority.
            threshold (float): The threshold for the priority decay.
            decay (float): The decay of the priority.
            p_scale (float): The minimum ratio of a priority to drop to.
        """
        super().__init__(capacity, alpha, beta, beta_increment, epsilon)

        self.threshold = threshold
        self.decay = decay
        self.p_scale = p_scale
        self.window_size = 0 if threshold == 0 else np.floor(np.log(threshold)
                                                             / np.log(decay))

        # Decay priority
        for i in range(1, self.window_size + 1):
            decay_idx = current_index - i
            decay_prio = self.priorities.get_leaf(decay_idx)

            updated_prio = np.max(priority * np.pow(self.decay, i), decay_prio)

            self.priorities.set(updated_prio, decay_idx)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        batch, indices, is_weights = super().sample(size)

        # Transpose batch and sequence dimensions
        batch = {key: value.transpose(0, 1) for key, value in batch.items()}

        return batch, indices, is_weights

    def update_priority(self, index, error):
        """
        Updates the priority of the experience at the given index, using the
        maximum of error given and the scaled priority.

        index : The index of the experience
        error : The new error of the experience
        """
        priority = self._get_priority(error)
        current_priority = self.priorities.get_leaf(index)
        priority = np.max([priority, self.p_scale * current_priority])
        self.priorities.set(priority, index)