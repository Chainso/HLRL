import numpy as np
from typing import Any, Dict, Tuple

from .binary_sum_tree import BinarySumTree
from .per import PER

class CER(PER):
    """
    Combined Experience Replay, an experience replay buffer comprised of
    multiple separate experience replay buffers.
    """
    def __init__(self,
                 experience_replays: Tuple[PER, ...],
                 priorities: Tuple[float, ...]):
        """
        Creates the experience replay manager for multiple separate experience
        replays.

        Args:
            experience_replays: The experience replays to add and sample from.
        """
        self.experience_replays = experience_replays
        self.priorities = np.array(priorities)

    def __len__(self) -> int:
        """
        Returns the length of the smallest experience replay buffer.

        Returns:
            The length of the smallest experiences replay buffer.
        """
        return np.min([len(er) for er in self.experience_replays])

    def add(self, experience: Dict, replay_index: int) -> None:
        """
        Adds the given experience to the replay buffer at the index.

        Args:
            experience: The experience dictionary to add to the buffer.
            replay_index: The index of the replay buffer to add to.
        """
        self.experience_replays[replay_index].add(experience)
 
    def sample(self, size: int) -> Tuple[Any,
                                         Tuple[int, Tuple[int, ...]],
                                         Tuple[Any, ...]]:
        """
        Samples "size" number of experiences from one of the experience replays.

        Args:
            size: The number of experiences to sample.
        """
        assert(size > 0)

        er_priorities = self.priorities / self.priorities.sum()

        experience_replay_idx = np.random.choice(
            len(self.priorities), p = er_priorities
        )
        experience_replay = self.experience_replays[experience_replay_idx]

        batch, indices, is_weights = experience_replay.sample(size)

        indices = experience_replay_idx, indices

        return batch, indices, is_weights

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
        experience_replay_idx, indices = indices
        experience_replay = self.experience_replays[experience_replay_idx]

        experience_replay.update_priorities(indices, q_vals, q_targets)
