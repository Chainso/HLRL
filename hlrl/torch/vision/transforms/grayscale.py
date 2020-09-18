import torch

from hlrl.core.vision.transforms import Transform

class Grayscale(Transform):
    """rrr
    A grayscale transform using the ITU-R BT.709 standard.
    """
    def __init__(self):
        """
        Creates the transform with the weights needed to perform the operation.
        """
        super().__init__()
        self.weights = torch.FloatTensor([0.2125, 0.7154, 0.0721])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "nchw, c -> nhw", tensor, self.weights.to(tensor.device)
        ).unsqueeze(1)