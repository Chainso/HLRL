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
        raise NotImplementedError

    @abstractmethod
    def add(self, *add_args):
        """
        Adds the given experience to the replay buffer.

        Args:
            add_args (tuple): The arguments to add the experience to the buffer
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        Args:
            size (int): The number of experiences to sample.
        """
        raise NotImplementedError