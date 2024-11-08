import torch
from .Activation import Activation


class Tanh(Activation):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, **kwargs)
        self.name = "Tanh"

    def forward(self, input, **kwargs):
        self.input = input
        output = torch.tanh(input)
        return output
    
    def backward(self, dCdy, **kwargs):
        dCdx = dCdy * (1 - torch.tanh(self.input) ** 2)
        return dCdx
