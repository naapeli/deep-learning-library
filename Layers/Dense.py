import torch
from Layers.Base import Base
from math import sqrt


class Dense(Base):
    def __init__(self, output_size, input_size=1, activation=None, **kwargs):
        super().__init__(output_size, input_size, activation)
        self.name = "Dense"

    def initialise_layer(self):
        # initialise parameters by Xavier initialisation
        self.weights = torch.normal(mean=0, std=sqrt(1/(self.input_size + self.output_size)), size=(self.input_size, self.output_size), dtype=self.data_type, device=self.device, requires_grad=False)
        self.biases = torch.zeros(self.output_size, dtype=self.data_type, device=self.device, requires_grad=False)
        self.nparams = self.output_size * self.input_size + self.output_size

    def forward(self, input, **kwargs):
        self.input = input
        self.output = self.input @ self.weights + self.biases
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, learning_rate=0.001, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy, learning_rate=learning_rate)
        dCdx = dCdy @ self.weights.T
        self.weights -= learning_rate * self.input.T @ dCdy
        self.biases -= learning_rate * torch.mean(dCdy, axis=0)
        return dCdx
