import torch
from torch import nn as nn

class IReLU(nn.Module):
    def __init__(self, low=-1, high=1):
        super().__init__()

        self.low = low
        self.high = high

    def forward(self, x):
        intervals = torch.linspace(
            self.low, self.high, x.shape[-1], device=x.device
        ).unsqueeze(0).expand(x.shape[0], -1)
        mask = x < intervals
 
        return  mask * intervals + ~mask * x