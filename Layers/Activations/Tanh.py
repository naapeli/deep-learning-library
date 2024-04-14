import numpy as np
from Layers.Base import Base


class Tanh(Base):
    def __init__(self, output_size=1, input_size=1, activation=None):
        assert activation == None, "Activation must be None on activation layers"
        super().__init__(output_size, output_size)
        self.name = "Tanh"

    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001):
        dCdx = dCdy * (1 - np.tanh(self.input) ** 2)
        return dCdx
