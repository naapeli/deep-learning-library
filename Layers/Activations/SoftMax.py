import torch
from Layers.Activations.Activation import Activation


class SoftMax(Activation):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size)
        self.name = "softMax"

    def forward(self, input, training=False):
        self.input = input
        exponential_input = torch.exp(self.input)
        self.output = exponential_input / torch.sum(exponential_input)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, training=False):
        n = self.output.shape[0]
        output_repeated = torch.tile(self.output, (n, 1))
        dCdx = (output_repeated.T * (torch.eye(n) - output_repeated)) @ dCdy
        return dCdx
