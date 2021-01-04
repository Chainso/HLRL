from typing import Dict

import torch
import numpy as np

from hlrl.core.experience_replay import PER

class TorchPER(PER):
    """
    A Prioritized Experience Replay implementation using torch tensors
    https://arxiv.org/abs/1511.05952
    """
    def get_error(self,
                  q_val: torch.Tensor,
                  q_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the error (absolute difference) between the Q-value and the
        target Q-value.

        Args:
            q_val: The Q-value of the experience.
            q_target: The target Q-value.

        Returns:
            The absolute difference between the Q-value and its target.
        """
        return torch.abs(q_val - q_target)

    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Adds the given experience to the replay buffer with the priority being
        the given error added to the epsilon value.

        Args:
            experience: The experience dictionary to add to the buffer.
        """
        q_val = experience.pop("q_val")
        target_q_val = experience.pop("target_q_val")

        error = self.get_error(q_val, target_q_val).item()

        current_index = self.priorities.next_index()
    
        # Store individually for faster "zipping"
        for key in experience:
            # Reuse tensors if possible to save storage space
            # ex. state for this experience may be the next_state of the last
            # experience
            # Otherwise need to clone so sender can remove their reference
            # Only possible to reuse since experiences are not modified after
            # insertion
            store_exp = None
            reuse = False
            
            if len(self) > 0:
                for old_key in self.experiences:
                    last_exp = self.experiences[old_key][current_index - 1]
     
                    if experience[key].equal(last_exp):
                        store_exp = last_exp
                        reuse = True
                        break

            if not reuse:
                store_exp = experience[key].clone()

            if key not in self.experiences:
                self.experiences[key] = np.zeros(self.capacity, dtype=object)

            self.experiences[key][current_index] = store_exp

        priority = self.get_priority(error)

        self.priorities.add(priority)

    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        size : The number of experiences to sample
        """
        assert(size > 0)

        priorities = self.priorities.get_leaves() / self.priorities.sum()
        priorities /= priorities.sum()

        indices = np.random.choice(len(priorities), size, p = priorities)

        batch = {}
        device = "cpu"
        for key in self.experiences:
            value = torch.cat(self.experiences[key][indices].tolist())

            # Cloning before sending prevents FD for CPU, not necessary for CUDA
            # since CUDA memory is inherently shared
            device = value.device
            if str(device) == "cpu":
                value = value.clone()

            batch[key] = value

        probabilities = priorities[indices]

        is_weights = np.power(len(self.priorities) * probabilities,
                              -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights).to(device)

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
        errors = self.get_error(q_vals, q_targets)

        for index, error in zip(indices, errors):
            index = index.item()
            error = error.item()

            self.update_priority(index, error)