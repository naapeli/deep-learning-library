import torch
from Layers.Base import Base
from math import sqrt


class Dense(Base):
    def __init__(self, output_shape, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape, activation=activation, normalisation=normalisation)
        self.name = "Dense"

    def initialise_layer(self):
        # initialise parameters by Xavier initialisation
        self.weights = torch.normal(mean=0, std=sqrt(1/(self.input_shape + self.output_shape)), size=(self.input_shape, self.output_shape), dtype=self.data_type, device=self.device)
        self.biases = torch.zeros(self.output_shape, dtype=self.data_type, device=self.device)
        self.nparams = self.output_shape * self.input_shape + self.output_shape

    def forward(self, input, training=False, **kwargs):
        self.input = input
        self.output = self.input @ self.weights + self.biases
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, learning_rate=0.001, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy, learning_rate=learning_rate)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy, learning_rate=learning_rate)
        dCdx = dCdy @ self.weights.T
        self.weights -= learning_rate * self.input.T @ dCdy
        self.biases -= learning_rate * torch.mean(dCdy, axis=0)
        return dCdx
