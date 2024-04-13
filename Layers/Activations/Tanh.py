import numpy as np
from Layers.Base import Base


class Tanh(Base):
    def __init__(self, input_size):
        super().__init__(input_size, input_size)

    def forward(self, input):
        self.input = input
        return np.tanh(input)
    
    def backward(self, dCdy, learning_rate=0.001):
        self.dCdx = dCdy * (1 - np.tanh(self.input) ** 2)
        return self.dCdx
