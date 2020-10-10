import torch

from typing import Callable

def initialize_weights(init_func: Callable[[torch.Tensor], None]):
    """
    Returns a function that can be applied to a module to initializes the module
    weights.

    Args:
        init_func (Callable[[torch.Tensor], None]): A function taking a tensor
            and modifies the tensor's weight data in place.
    """
    def applied_func(module):
        """
        The function that is applied to the module to initialize the weights
        """
        if hasattr(module, "weight"):
            init_func(module.weight.data)

    return initialize_weights