from typing import Iterable

from gymnasium import Space
from gymnasium.spaces import Tuple

class TupleShaped(Tuple):
    """
    A tuple space, with the shape being the shapes of the underlying spaces
    concatenated.
    """
    def __init__(self, spaces: Iterable[Space]):
        """
        Creates the tuple space around the spaces.

        Args:
            spaces: The spaces of this tuple.
        """
        super().__init__(spaces)

        self._shape = tuple(space.shape for space in spaces)

class FlattenedTupleShaped(TupleShaped):
    """
    A tuple space where the shape is flattened into a single dimension.
    """
    def __init__(self, spaces: Iterable[Space]):
        """
        Creates the tuple space around the spaces.

        Args:
            spaces: The spaces of this tuple.
        """
        super().__init__(spaces)

        self._shape = sum(self.shape, tuple())
