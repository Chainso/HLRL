import queue
from abc import ABC, abstractmethod
from typing import Any, Iterable


class ExperienceReplay(ABC):
    """
    An abstract replay buffer.
    """
    def __init__(self, capacity: int):
        """
        Creates the replay buffer of the capacity given.

        Args:
            capacity: The capacity of the replay buffer.
        """
        self.capacity = capacity

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of experiences added to the buffer.

        Returns:
            The number of experiences added.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, args: Any) -> None:
        """
        Adds the given experience to the replay buffer.

        Args:
            args: The arguments to add the experience to the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, size: int) -> Dict[str, Iterable[Any]]:
        """
        Samples "size" number of experiences from the buffer.

        Args:
            size: The number of experiences to sample.
        
        Returns:
            A sample of the size given from the replay buffer.
        """
        raise NotImplementedError
