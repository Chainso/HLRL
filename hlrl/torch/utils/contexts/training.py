from contextlib import contextmanager

import torch.nn as nn

@contextmanager
def evaluate(module: nn.Module):
    """
    A context manager for evaluating the module.

    Args:
        module: The module to switch to evaluating in the context.

    Returns:
        A generator for the context of the module.
    """
    training = module.training

    try:
        module.eval()
        yield module
    finally:
        # Switch batch to training if needed
        if training:
            module.train()
