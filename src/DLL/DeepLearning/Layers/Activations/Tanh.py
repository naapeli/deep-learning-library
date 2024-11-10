import torch
from .Activation import Activation


class Tanh(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Tanh"

    def forward(self, input, **kwargs):
        self.input = input
        output = torch.tanh(input)
        return output
    
    def backward(self, dCdy, **kwargs):
        dCdx = dCdy * (1 - torch.tanh(self.input) ** 2)
        return dCdx
