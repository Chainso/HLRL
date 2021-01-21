from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .binary_sum_tree import BinarySumTree
from .replay import ExperienceReplay

class PER(ExperienceReplay):
    """
    A Prioritized Experience Replay implementation.
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self,
                 capacity: int,
                 alpha: float,
                 beta: float,
                 beta_increment: float,
                 epsilon: float):
        """
        Creates a new PER buffer with the given parameters.

        Args:
            capacity: The capacity of the replay buffer.
            alpha: The alpha value for the prioritization, between 0 and 1
                inclusive.
            beta: The beta value for the importance sampling, between 0 and 1
                inclusive.
            beta_increment: The value to increment the beta by.
            epsilon: The value of epsilon to add to the priority.
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.experiences = {}
        self.priorities = BinarySumTree(capacity)
        self.ids = np.zeros(self.capacity, dtype=np.object)

    def __len__(self) -> int:
        """
        Returns the number of experiences added to the buffer.

        Returns:
            The number of experiences added.
        """
        return len(self.priorities)

    def get_priority(self, error: np.array) -> np.array:
        """
        Computes the priority for the given error.

        Args:
            error: The error to get the priority for.
        
        Returns:
            The calculated priority using the error given.
        """
        return (error + self.epsilon) ** self.alpha

    def get_error(self, q_val: np.array, q_target: np.array) -> np.array:
        """
        Computes the error (absolute difference) between the Q-value plus the
        reward and the discounted Q-value of the next state.

        Args:
            q_val: The Q value of the experiences.
            q_target: The target Q-value of the experiences.

        Returns:
            The absolute difference between the Q-value and its target.
        """
        return np.abs(q_val - q_target)

    def add(self, experience: Dict[str, Any], priority: float) -> None:
        """
        Adds the experience to the buffer with the priority given.

        Args:
            experience: The experience to add.
            priority: The priority of the experience.
        """
        current_index = self.priorities.next_index()

        # Store individually for faster "zipping"
        for key in experience:
            if key not in self.experiences:
                self.experiences[key] = np.zeros(self.capacity, dtype=np.object)

            self.experiences[key][current_index] = experience[key]

        # Store the ID in order to check if replay is still in the buffer
        exp_id = None
        if "id" in self.experiences[key][current_index]:
            exp_id = self.experiences[key][current_index].pop("id")

        self.ids[current_index] = exp_id
        self.priorities.add(priority)
        
    def calculate_and_add(self, experience: Dict[str, Any]) -> None:
        """
        Adds the given experience to the replay buffer with the priority being
        the given calculated from the the Q-value and target Q-value, added to
        the epsilon value.

        Args:
            experience: The experience dictionary to add to the buffer.
        """
        q_val = experience.pop("q_val")
        target_q_val = experience.pop("target_q_val")

        error = self.get_error(q_val, target_q_val)
        priority = self.get_priority(error).item()
    
        self.add(experience, priority)

    def sample(
            self,
            size: int
        ) -> Tuple[Dict[str, np.array], Tuple[np.array, Tuple[Any]], np.array]:
        """
        Samples "size" number of experiences from the buffer.

        Args:
            size: The number of experiences to sample.
        
        Returns:
            A sample of the size given from the replay buffer along with the
            identifier to update the priorities of the experiences and the
            importance sampling weights.
        """
        priorities = self.priorities.get_leaves() / self.priorities.sum()

        # A hack right now
        #priorities /= priorities.sum()

        indices = np.random.choice(len(priorities), size, p = priorities)

        ids = self.ids[indices]
        ids = tuple(zip(indices, ids))

        batch = {
            key: np.concatenate(self.experiences[key][indices].tolist())
            for key in self.experiences
        }

        probabilities = priorities[indices]

        is_weights = np.power(len(self) * probabilities, -self.beta)
        is_weights /= is_weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return batch, ids, is_weights

    def update_priority(self, index: int, priority: float) -> None:
        """
        Updates the priority of the experience at the index to the one given.

        Args:
            index: The index of the experience.
            priority: The updated priority of the experience:
        """
        self.priorities.set(priority, index)

    def update_priorities(
            self,
            ids: Tuple[Iterable[int], Iterable[Any]],
            priorities: np.array
        ) -> None:
        """
        Updates the priority of the experiences at the given indices.

        Args:
            ids: The indices and ids of the experiences.
            priorities: The updated priorities of the experiences.
        """
        for (index, exp_id), priority in zip(ids, priorities):
            if self.ids[index] == exp_id:
                self.update_priority(index, priority.item())

    def calculate_and_update_priorities(
            self,
            ids: Tuple[Iterable[int], Iterable[Any]],
            q_vals: np.array,
            q_targets: np.array
        ) -> None:
        """
        Updates the priority of the experiences at the given indices, using the
        errors given.

        Args:
            ids: The indices and ids of the experiences.
            q_vals: The updated Q-values of the actions taken.
            q_targets: The new targets for the Q-values.
        """
        errors = self.get_error(q_vals, q_targets)
        priorities = self.get_priority(errors)

        self.update_priorities(ids, priorities)
