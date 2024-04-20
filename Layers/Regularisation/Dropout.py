import torch
from Layers.Activations.Activation import Activation


class Dropout(Activation):
    def __init__(self, output_size=None, p=0.5, **kwargs):
        super().__init__(output_size)
        self.p = 1 - p
        self.mask = torch.rand(self.output_size, dtype=self.data_type, requires_grad=False, device=self.device) < self.p
        self.name = "Dropout"

    def forward(self, input, training=False, **kwargs):
        self.input = input
        if training:
            self.mask = torch.rand(size=self.input.shape, dtype=self.data_type, requires_grad=False, device=self.device) < self.p
            self.output = self.input * self.mask / self.p
        else:
            self.output = self.input
        return self.output
    
    def backward(self, dCdy, training=False, **kwargs):
        dCdx = dCdy * self.mask if training else dCdy
        return dCdx
    
    def summary(self):
        return f"{self.name} - Output: ({self.output_size}) - Keep probability: {self.p}"
