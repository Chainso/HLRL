from typing import Any, Dict, Tuple

import torch
import numpy as np

from hlrl.core.experience_replay import PER

class TorchPER(PER):
    """
    A Prioritized Experience Replay implementation using torch tensors
    https://arxiv.org/abs/1511.05952
    """
    def get_priority(self, error: torch.Tensor) -> np.array:
        """
        Computes the priority for the given error.

        Args:
            error: The error to get the priority for.
        
        Returns:
            The calculated priority using the error given.
        """
        return super().get_priority(error).cpu().numpy()

    def get_error(
            self,
            q_val: torch.Tensor,
            q_target: torch.Tensor
        ) -> torch.Tensor:
        """
        Computes the error (absolute difference) between the Q-value plus the
        reward and the discounted Q-value of the next state.

        Args:
            q_val: The Q value of the experiences.
            q_target: The target Q-value of the experiences.

        Returns:
            The absolute difference between the Q-value and its target.
        """
        return torch.abs(q_val - q_target)

    def sample(
            self,
            size: int
        ) -> Tuple[Dict[str, torch.Tensor],
                   Tuple[np.array, Tuple[Any]],
                   torch.Tensor]:
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
        priorities /= priorities.sum()

        indices = np.random.choice(len(priorities), size, p = priorities)

        ids = self.ids[indices]
        ids = tuple(zip(indices, ids))

        batch = {}
        device = "cpu"

        for key in self.experiences:
            batch[key] = torch.cat(self.experiences[key][indices].tolist())
            device = batch[key].device

        probabilities = priorities[indices]

        is_weights = np.power(len(self) * probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights).to(device)

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return batch, ids, is_weights
