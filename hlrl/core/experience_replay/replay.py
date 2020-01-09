from abc import ABC, abstractmethod

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

    def get_from_queue(self, queue):
        """
        Retrieve samples from a queue.
        """
        self.add(*queue.get())

    @abstractmethod
    def sample(self, size):
        """
        Samples "size" number of experiences from the buffer

        Args:
            size (int): The number of experiences to sample.
        """
        pass

class ExperienceReplayQueue():
    """
    Wraps the experience replay by passing objects through a pipe.
    """
    def __init__(self, er):
        """
        Args:
            er (ExperienceReplay): The experience replay to wrap.
        """
        self.er = er

    def __getattr__(self, name):
        return getattr(self.wrappee, name)

    def add(self, name):
        """
        Adds by
        """
        pass