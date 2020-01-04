import numpy as np

from .binary_sum_tree import BinarySumTree

class PER():
    """
    A Prioritized Experience Replay implementation
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity: int, alpha: float, beta: float,
                 beta_increment: float, epsilon: float, device: str):
        """
        Creates a new PER buffer with the given parameters

        Args:
            capacity (int): The capacity of the replay buffer
            alpha (float): The alpha value for the prioritization,
                           between 0 and 1 inclusive
            beta (float): The beta value for the importance sampling,
                          between 0 and 1 inclusive
            beta_increment (float): The value to increment the beta by
            epsilon (float): The value of epsilon to add to the priority
            device (str): The device of the tensors in the replay buffer
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experiences = np.zeros(capacity, dtype=object)
        self.priorities = BinarySumTree(capacity)

    def __len__(self):
        """
        Returns the number of experiences added to the buffer
        """
        return len(self.priorities)

    def _get_priority(self, error):
        """
        Computes the priority for the given error

        error : The error to get the priority for
        """
        return (error + self.epsilon) ** self.alpha

    def add(self, experience, q_val, discounted_next_q):
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value.

        Args:
            experience (tuple) : The (s, a, r, s', t) experience to add to the
                                 buffer

            q_val (float): The Q-value of the action taken in the experience

            discounted_next_q (float): The discounted Q-value of the "optimal"
                                       next action
        """
        error = np.abs(q_val - discounted_next_q)

        current_index = self.priorities.next_index()
        self.experiences[current_index] = np.array(experience)

        priority = self._get_priority(error)
        self.priorities.add(priority)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        priorities = self.priorities.get_leaves() / self.priorities.sum()
        indices = np.random.choice(len(priorities), size, p = priorities)

        batch = np.stack(self.experiences[indices])

        stacked_batch = []

        # In order to get float32 instead of float64 and long over int
        for arr in batch.transpose():
            stacked_arr = np.stack(arr)

            if(stacked_arr.dtype == np.float64):
                stacked_arr = stacked_arr.astype(np.float32)
            elif(stacked_arr.dtype == np.int32):
                stacked_arr = stacked_arr.astype(np.int64)

            stacked_batch.append(stacked_arr)

        probabilities = priorities[indices]

        is_weights = np.power(len(self.priorities) * probabilities, -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return stacked_batch, indices, is_weights

    def update_priority(self, index, error):
        """
        Updates the priority of the experience at the given index, using the
        error given

        index : The index of the experience
        error : The new error of the experience
        """
        priority = self._get_priority(error)
        self.priorities.set(priority, index)

    def update_priorities(self, indices, q_vals, discounted_next_q):
        """
        Updates the priority of the experiences at the given indices, using the
        errors given.

        Args:
            q_val (float): The Q-value of the actions taken

            discounted_next_qs (float): The discounted Q-value of the "optimal"
                                       next actions
        """
        errors = np.abs(q_vals - discounted_next_q)

        for index, error in zip(indices, errors):
            self.update_priority(index, error)
