from .Base import Base
import numpy as np 


class Flatten(Base):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
        self.name = "Flatten"
    
    def initialise_layer(self):
        self.output_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))
    
    def forward(self, input, **kwargs):
        self.input = input
        return input.reshape(input.shape[0], -1)
    
    def backward(self, dCdy, **kwargs):
        return dCdy.reshape(*self.input.shape)
