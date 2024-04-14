import numpy as np
from Layers.Base import Base


class SoftMax(Base):
    def __init__(self, output_size=1, input_size=1, activation=None):
        assert activation == None, "Activation must be None on activation layers"
        super().__init__(output_size, output_size)
        self.name = "softMax"

    def forward(self, input):
        self.input = input
        exponential_input = np.exp(self.input)
        self.output = exponential_input / np.sum(exponential_input)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001):
        n = np.size(self.output)
        output_repeated = np.tile(self.output, (n, 1))
        dCdx = (output_repeated.T * (np.identity(n) - output_repeated)) @ dCdy
        return dCdx
