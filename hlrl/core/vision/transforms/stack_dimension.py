from torch import Tensor

from hlrl.core.vision.transforms import Transform

class StackDimension(Transform):
    """
    A transforms that stacks a batch of images along the channel dimension
    """
    def __init__(self, dimension: int):
        """
        Creates the transform that stacks along the dimension provided.

        Args:
            dimension (int): The dimension to stack the batch of images along.
        """
        super().__init__()

        self.dimension = dimension

    def __call__(self, tensor: Tensor) -> Tensor:
        batch_size = frame.shape[0]

        return frame.view(
            1, *frame.shape[1:self.dimension],
            batch_size * frame.shape[self.dimension],
            frame.shape[self.dimension + 1:]
        )