import torch
from Layers.Base import Base
from math import sqrt


class Dense(Base):
    def __init__(self, output_shape, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
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

    def backward(self, dCdy, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        dCdx = dCdy @ self.weights.T
        self.weights.grad = self.input.T @ dCdy
        self.biases.grad = torch.mean(dCdy, axis=0)
        return dCdx
    
    def get_parameters(self):
        return (self.weights, self.biases)
