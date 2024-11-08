import torch
from .BaseRegularisation import BaseRegularisation


class Dropout(BaseRegularisation):
    def __init__(self, output_shape=None, p=0.5, **kwargs):
        super().__init__(output_shape, **kwargs)
        assert 0 <= p and p <= 1, "Dropout probability must be between 0 and 1"
        self.p = 1 - p
        self.name = "Dropout"
    
    def initialise_layer(self, **kwargs):
        super().initialise_layer(**kwargs)
        self.mask = torch.rand(size=self.output_shape, dtype=self.data_type, device=self.device) < self.p

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
        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        return f"{self.name} - Output: ({output_shape}) - Keep probability: {self.p}"
