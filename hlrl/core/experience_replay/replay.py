import queue

from abc import ABC, abstractmethod
from torch.multiprocessing import JoinableQueue


class ExperienceReplay(ABC):
    """
    An abstract replay buffer
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity (int): The capacity of the replay buffer
        """
        self.capacity = capacity

    @abstractmethod
    def __len__(self):
        """
        Returns the number of experiences added to the buffer.
        """
        pass

    @abstractmethod
    def add(self, experience):
        """
        Adds the given experience to the replay buffer.

        Args:
            experience (tuple) : The (s, a, r, ...) experience to add to the
                                 buffer.
        """
        pass

    @abstractmethod
    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        Args:
            size (int): The number of experiences to sample.
        """
        pass