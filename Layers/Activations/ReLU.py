import numpy as np
from Layers.Activations.Activation import Activation


class ReLU(Activation):
    def __init__(self, output_size=1, input_size=None, activation=None):
        super().__init__(output_size, input_size)
        self.name = "ReLU"

    def forward(self, input):
        self.input = input
        self.output = np.maximum(self.input, 0)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001):
        dCdx = dCdy * (self.input > 0)
        return dCdx
