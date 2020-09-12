from torch import Tensor
from torch.nn.functional import interpolate
from typing import Optional, Union, Tuple

from hlrl.core.vision.transforms import Transform

class Interpolate(Transform):
    """
    Uses torch's interpolate function to resize images.
    """
    def __init__(self,
        size: Optional[Union[int, Tuple[int], Tuple[int, int],
            Tuple[int, int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float]]] = None,
        mode: str = 'nearest', align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None):
        """
        Creates the interpolate transform, performing the same function as
        pytorch's torch.nn.interpolate.

        Args:
            size (Optional[Union[int, Tuple[int], Tuple[int, int],
                Tuple[int, int, int]]]): The size of the output.
            scale_factor (Optional[Union[float, Tuple[float]]]): Multiplier for
                spatial size. Has to match input size if it is a tuple.
            mode (str): Algorithm used for upsampling.
            align_pixels (Optional[bool]): If true, aligns tensors by the center
                of the corner pixels, perserving the values of the corners.
            recompute_scale_factor (Optional[bool]): Will recalculate the scale
                factor based on the input and output sizes if true.
        """
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def __call__(self, tensor: Tensor) -> Tensor:
        return interpolate(
            tensor, self.size, self.scale_factor, self.mode, self.align_corners,
            self.recompute_scale_factor
        )