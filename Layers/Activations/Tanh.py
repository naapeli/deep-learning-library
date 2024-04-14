import numpy as np
from Layers.Activations.Activation import Activation


class Tanh(Activation):
    def __init__(self, output_size=None, input_size=None, activation=None):
        super().__init__(output_size, input_size)
        self.name = "Tanh"

    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001):
        dCdx = dCdy * (1 - np.tanh(self.input) ** 2)
        return dCdx
