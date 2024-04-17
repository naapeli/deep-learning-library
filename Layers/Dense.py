import torch
from Layers.Base import Base
from math import sqrt


class Dense(Base):
    def __init__(self, output_size, input_size=1, activation=None):
        super().__init__(output_size, input_size, activation)
        # initialise parameters by Xavier initialisation
        self.weights = torch.normal(mean=0, std=sqrt(1/(input_size + output_size)), size=(output_size, input_size), dtype=torch.float32, device=self.device, requires_grad=False)
        self.biases = torch.zeros(size=(output_size, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.name = "Dense"
        self.nparams = output_size * input_size + output_size

    def forward(self, input):
        self.input = input
        self.output = self.weights @ self.input + self.biases
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, learning_rate=0.001):
        if self.activation: dCdy = self.activation.backward(dCdy, learning_rate)
        dCdx = self.weights.T @ dCdy
        self.weights -= learning_rate * dCdy @ self.input.T
        self.biases -= learning_rate * torch.mean(dCdy, axis=1, keepdims=True)
        return dCdx
