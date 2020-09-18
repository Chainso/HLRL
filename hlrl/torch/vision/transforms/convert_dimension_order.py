from torch import Tensor

from hlrl.torch.vision.transforms import Transform

class ConvertDimensionOrder(Transform):
    """
    Converts a tensor of shape (N)HWC -> (N)CHW
    """
    def __call__(self, tensor: Tensor) -> Tensor:
        if len(tensor.shape) == 4:
            ret = tensor.permute(0, 3, 1, 2)
        else:
            ret = tensor.permute(2, 0, 1)

        ret = ret.contiguous()

        return ret