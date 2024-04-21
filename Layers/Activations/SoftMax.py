import torch
from Layers.Activations.Activation import Activation


class SoftMax(Activation):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size)
        self.name = "Softmax"

    def forward(self, input, **kwargs):
        self.input = input
        exponential_input = torch.exp(self.input - torch.max(self.input, dim=1, keepdim=True).values)
        self.output = exponential_input / torch.sum(exponential_input, dim=1, keepdim=True)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        n = self.output_size
        dCdx = torch.stack([(dCdy[i] @ (torch.tile(datapoint, (n, 1)).T * (torch.eye(n) - torch.tile(datapoint, (n, 1))))) for i, datapoint in enumerate(self.output)])
        return dCdx
