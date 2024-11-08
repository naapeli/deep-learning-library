import torch
from .Activation import Activation


class ReLU(Activation):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, **kwargs)
        self.name = "ReLU"

    def forward(self, input, **kwargs):
        self.input = input
        output = torch.maximum(self.input, torch.zeros_like(self.input, device=self.input.device, dtype=self.input.dtype))
        return output
    
    def backward(self, dCdy, **kwargs):
        dCdx = dCdy * (self.input > 0)
        return dCdx
