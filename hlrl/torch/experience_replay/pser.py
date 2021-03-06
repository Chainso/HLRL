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

        current_index = self.priorities.next_index()

        # Decay priority
        for i in range(1, self.window_size + 1):
            decay_idx = current_index - i
            decay_prio = self.priorities.get_leaf(decay_idx)

            updated_prio = np.max(priority * np.pow(self.decay, i), decay_prio)

            self.priorities.set(updated_prio, decay_idx)


    def get_priority(self, error):
        """
        Computes the priority for the given error

        error : The error to get the priority for
        """
        priority = super().get_priority(error)
        current_priority = self.priorities.get_leaf(index)

        priority = np.max([priority, self.p_scale * current_priority])

        return priority
