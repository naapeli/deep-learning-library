import torch
from Layers.Activations.Activation import Activation


class Dropout(Activation):
    def __init__(self, output_size=None, p=0.5, **kwargs):
        super().__init__(output_size)
        self.p = p
        self.mask = torch.rand(self.output_size, dtype=self.data_type) < self.p
        self.name = "Dropout"

    def forward(self, input, training=False):
        self.input = input
        if training:
            self.mask = torch.rand(size=self.input.shape) < self.p
            self.output = self.input * self.mask / self.p
        else:
            self.output = self.input
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, training=False):
        dCdx = dCdy * self.mask if training else dCdy
        return dCdx
    
    def summary(self):
        return f"{self.name} - Output: ({self.output_size}) - Keep probability: {self.p}"
