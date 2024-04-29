import torch
from Layers.Activations.Activation import Activation


class Tanh(Activation):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape)
        self.name = "Tanh"

    def forward(self, input, **kwargs):
        self.input = input
        self.output = torch.tanh(input)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        dCdx = dCdy * (1 - torch.tanh(self.input) ** 2)
        return dCdx
