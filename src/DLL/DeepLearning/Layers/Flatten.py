import numpy as np

from .BaseLayer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
        self.name = "Flatten"
    
    def initialise_layer(self, input_shape, data_type, device):
        super().initialise_layer(input_shape, data_type, device)
        self.output_shape = (np.prod(self.input_shape),)
    
    def forward(self, input, **kwargs):
        self.input = input
        return input.reshape(input.shape[0], -1)
    
    def backward(self, dCdy, **kwargs):
        return dCdy.reshape(*self.input.shape)
