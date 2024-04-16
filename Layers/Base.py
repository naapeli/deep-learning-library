import numpy as np


class Base:
    def __init__(self, output_size, input_size=None, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.output = None
        self.input = None
        self.nparams = 0
        self.name = "base"
        self.activation = activation
        if self.activation: self.activation.__init__(output_size, output_size)

    def summary(self):
        return f"{self.name}: ({self.input_size}, {self.output_size}) - {self.nparams}" + (" - Activation: " + self.activation.name if self.activation else "")

    def forward(self, input):
        self.input = input.T
        return self.input

    def backward(self, dCdy, learning_rate=0.001):
        return dCdy
