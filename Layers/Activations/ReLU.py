import torch
from Layers.Activations.Activation import Activation


class ReLU(Activation):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size)
        self.name = "ReLU"

    def forward(self, input, training=False):
        self.input = input
        self.output = torch.maximum(self.input, torch.zeros_like(self.input))
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, training=False):
        dCdx = dCdy * (self.input > 0)
        return dCdx
