import numpy as np
from Layers.Base import Base


class Dense(Base):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.weights = np.random.normal(size=(output_size, input_size))
        self.biases = np.random.normal(size=(output_size, 1))

    def forward(self, input):
        self.input = input
        return self.weights @ self.input + self.biases

    def backward(self, dCdy, learning_rate=0.001):
        self.dCdx = self.weights.T @ dCdy
        self.weights -= learning_rate * dCdy @ self.input.T
        self.biases -= learning_rate * dCdy
        return self.dCdx
