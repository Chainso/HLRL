from torch import Tensor

from hlrl.torch.vision.transforms import Transform

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
        batch_size = tensor.shape[0]

        return tensor.view(
            1, *tensor.shape[1:self.dimension],
            batch_size * tensor.shape[self.dimension],
            *tensor.shape[self.dimension + 1:]
        )