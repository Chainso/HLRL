from gym.spaces import Discrete

class DiscreteShaped(Discrete):
    """
    A tuple space, with the shape being a 1-dimensional tuple of one element
    containing the number of discrete actions.
    """
    def __init__(self, n):
        """
        Creates the discrete space with a space.

        Args:
            n: The number of discrete elements in this space.
        """
        super().__init__(n)

        self.shape = (n,)
