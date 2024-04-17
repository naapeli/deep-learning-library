import torch
from Layers.Base import Base
from math import sqrt


class Dense(Base):
    def __init__(self, output_size, input_size=1, activation=None, **kwargs):
        super().__init__(output_size, input_size, activation)
        self.name = "Dense"

    def initialise_layer(self):
        # initialise parameters by Xavier initialisation
        self.weights = torch.normal(mean=0, std=sqrt(1/(self.input_size + self.output_size)), size=(self.output_size, self.input_size), dtype=self.data_type, device=self.device, requires_grad=False)
        self.biases = torch.zeros(size=(self.output_size, 1), dtype=self.data_type, device=self.device, requires_grad=False)
        self.nparams = self.output_size * self.input_size + self.output_size

    def forward(self, input, training=False):
        self.input = input
        self.output = self.weights @ self.input + self.biases
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, learning_rate=0.001, training=False):
        if self.activation: dCdy = self.activation.backward(dCdy, learning_rate)
        dCdx = self.weights.T @ dCdy
        self.weights -= learning_rate * dCdy @ self.input.T
        self.biases -= learning_rate * torch.mean(dCdy, axis=1, keepdims=True)
        return dCdx
