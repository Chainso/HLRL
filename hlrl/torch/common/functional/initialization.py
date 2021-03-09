from typing import Callable

import torch
from torch import nn

def initialize_weights(
        init_func: Callable[[torch.Tensor], None]
    ) -> Callable[[nn.Module], None]:
    """
    Returns a function that can be applied to a module to initializes the module
    weights.

    Args:
        init_func: A function taking a torch module and modifies the module's
        weight data in place.
    """
    def applied_func(module):
        """
        The function that is applied to the module to initialize the weights
        """
        if hasattr(module, "weight"):
            init_func(module.weight.data)

    return applied_func

def initialize_bias(
        init_func: Callable[[torch.Tensor], None]
    ) -> Callable[[nn.Module], None]:
    """
    Returns a function that can be applied to a module to initializes the module
    bias.

    Args:
        init_func: A function taking a torch module and modifies the module's
        bias data in place.
    """
    def applied_func(module):
        """
        The function that is applied to the module to initialize the weights
        """
        if hasattr(module, "bias"):
            init_func(module.bias.data)

    return applied_func
