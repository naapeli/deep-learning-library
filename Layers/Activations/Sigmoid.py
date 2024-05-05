import torch
from Layers.Activations.Activation import Activation


class Sigmoid(Activation):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, **kwargs)
        self.name = "Sigmoid"

    def forward(self, input, **kwargs):
        self.input = input
        self.output = 1 / (1 + torch.exp(-self.input))
        return self.output
    
    def backward(self, dCdy, **kwargs):
        dCdx = (self.output * (1 - self.output)) * dCdy
        return dCdx
