import numpy as np
from Layers.Base import Base


class Dense(Base):
    def __init__(self, output_size, input_size=1):
        super().__init__(output_size, input_size)
        # initialise parameters by random numbers between -1 and 1
        self.weights = np.random.uniform(size=(output_size, input_size)) * 2 - 1
        self.biases = np.random.uniform(size=(output_size, 1)) * 2 - 1
        self.name = "Dense"
        self.nparams = output_size * input_size + output_size

    def forward(self, input):
        self.input = input
        self.output = self.weights @ self.input + self.biases
        return self.output

    def backward(self, dCdy, learning_rate=0.001):
        dCdx = self.weights.T @ dCdy
        self.weights -= learning_rate * dCdy @ self.input.T
        self.biases -= learning_rate * dCdy
        return dCdx
