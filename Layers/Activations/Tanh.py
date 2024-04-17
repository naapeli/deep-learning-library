import torch
from Layers.Activations.Activation import Activation


class Tanh(Activation):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size)
        self.name = "Tanh"

    def forward(self, input, training=False):
        self.input = input
        self.output = torch.tanh(input)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, training=False):
        dCdx = dCdy * (1 - torch.tanh(self.input) ** 2)
        return dCdx
