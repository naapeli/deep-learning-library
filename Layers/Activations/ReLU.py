import numpy as np
from Layers.Base import Base


class ReLU(Base):
    def __init__(self, input_size):
        super().__init__(input_size, input_size)

    def forward(self, input):
        self.input = input
        return np.maximum(self.input, 0)
    
    def backward(self, dCdy, learning_rate=0.001):
        self.dCdx = dCdy * (self.input > 0)
        return self.dCdx
