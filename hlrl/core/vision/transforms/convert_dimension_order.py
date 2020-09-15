from torch import Tensor

from hlrl.core.vision.transforms import Transform

class ConvertDimensionOrder(Transform):
    """
    Converts a tensor of shape NHWC -> NCHW
    """
    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.permute(0, 3, 1, 2).contiguous()