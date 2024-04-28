import torch
from Layers.Base import Base


class Flatten(Base):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
        self.name = "Flatten"
    
    def forward(self, input, **kwargs):
        self.input = input
        return input.reshape(input.shape[0], -1)
    
    def backward(self, dCdy, **kwargs):
        return dCdy.reshape(*self.input.shape)
