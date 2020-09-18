from torch import Tensor

from hlrl.core.vision.transforms import Transform

class MakeBatch(Transform):
    """
    Creates a batch of length one from a single tensor.
    """
    def __init__(self, dimension: int):
        """
        Creates the transform that creates the batch.

        Args:
            dimension (int): The dimension to make the batch in
        """
        super().__init__()

        self.dimension = dimension

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(self.dimension)