import torch
from Layers.Activations.Activation import Activation


class Dropout(Activation):
    def __init__(self, output_shape=None, p=0.5, **kwargs):
        super().__init__(output_shape, **kwargs)
        assert 0 <= p and p <= 1, "Dropout probability must be between 0 and 1"
        self.p = 1 - p
        self.name = "Dropout"
    
    def initialise_layer(self):
        self.output_shape = self.input_shape
        self.mask = torch.rand(size=self.output_shape[1:], dtype=self.data_type, device=self.device) < self.p

    def forward(self, input, training=False, **kwargs):
        self.input = input
        if training:
            self.mask = torch.rand(size=self.input.shape, dtype=self.data_type, device=self.device) < self.p
            self.output = self.input * self.mask / self.p
        else:
            self.output = self.input
        return self.output
    
    def backward(self, dCdy, training=False, **kwargs):
        dCdx = dCdy * self.mask if training else dCdy
        return dCdx
    
    def summary(self):
        return f"{self.name} - Output: ({self.output_shape}) - Keep probability: {self.p}"
