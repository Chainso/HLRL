import torch

from torch.nn.functional import interpolate
from typing import Optional, Union, Tuple

from hlrl.core.vision.transforms import Transform

class ConvertDimensionOrder(Transform):
    """
    Converts a tensor of shape NHWC -> NCHW
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 3, 1, 2)