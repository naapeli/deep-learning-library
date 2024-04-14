import numpy as np


class Base:
    def __init__(self, output_size, input_size=1):
        self.input_size = input_size
        self.output_size = output_size
        self.output = np.zeros(shape=(output_size, 1))
        self.input = None
        self.nparams = 0
        self.name = "base"

    def summary(self):
        return f"{self.name}: ({self.input_size}, {self.output_size}) - {self.nparams}"

    def forward(self, input):
        self.input = input.reshape(-1, 1)
        return self.input

    def backward(self, dCdy, learning_rate=0.001):
        return dCdy
