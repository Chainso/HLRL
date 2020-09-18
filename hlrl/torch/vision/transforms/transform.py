from abc import abstractmethod
from torch import Tensor

class Transform():
    """
    A generic transform to be used with the frame handler.
    """
    @abstractmethod
    def __call__(self, tensor: Tensor):
        """
        Applied just like any other unary function to the tensor.

        Args:
            tensor (Tensor): The tensor to transform.
        """
        raise NotImplementedError