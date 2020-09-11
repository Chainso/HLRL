import torch

from torch.nn.functional import interpolate
from typing import Optional, Union, Tuple

from hlrl.core.vision.transforms import Transform

class Grayscale(Transform):
    """
    A generic transform to be used with the frame handler.
    """
    def __init__(self):
        """
        Calculates grayscale using the ITU-R BT.709 standard.
        """
        super().__init__(self)
        self.weights = torch.FloatTensor([0.2125, 0.7154, 0.0721])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.matmul(tensor, self.weights)