import torch
from math import sqrt
from Layers.Activations.Activation import Activation


class BatchNorm1d(Activation):
    def __init__(self, output_size=None, patience=0.9, **kwargs):
        super().__init__(output_size)
        self.gamma = torch.ones(self.output_size, dtype=self.data_type, device=self.device, requires_grad=False)
        self.beta = torch.zeros(self.output_size, dtype=self.data_type, device=self.device, requires_grad=False)
        self.running_var = torch.ones(self.output_size, dtype=self.data_type, device=self.device, requires_grad=False)
        self.running_mean = torch.zeros(self.output_size, dtype=self.data_type, device=self.device, requires_grad=False)
        self.x_norm = None
        self.x_centered = None
        self.std = None
        self.patience = patience
        self.epsilon = 1e-6
        self.name = "BatchNormalisation"

    def forward(self, input, training=False, **kwargs):
        self.input = input
        if training:
            # mean and variance across the batch
            mean = torch.mean(input, axis=0)
            variance = torch.var(input, axis=0)
            self.std = torch.sqrt(variance + self.epsilon)
            self.running_mean = self.patience * self.running_mean + (1 - self.patience) * mean
            self.running_var = self.patience * self.running_var + (1 - self.patience) * variance
            self.x_centered = (self.input - mean)
            self.x_norm = self.x_centered / self.std
            self.output = self.gamma * self.x_norm + self.beta
        else:
            self.output = self.gamma * ((self.input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)) + self.beta
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, **kwargs):
        batch_size = self.output.shape[0]
        dCdx_norm = dCdy * self.gamma
        dCdgamma = (dCdy * self.x_norm).sum(axis=0)
        dCdbeta = dCdy.sum(axis=0)
        dCdvar = (dCdx_norm * self.x_centered * -self.std**(-3) / 2).sum(axis=0)
        dCdmean = -((dCdx_norm / self.std).sum(axis=0) + dCdvar * (2 / batch_size) * self.x_centered.sum(axis=0))
        dCdx = dCdx_norm / self.std + (dCdvar * 2 * self.x_centered + dCdmean) / batch_size

        self.gamma -= learning_rate * dCdgamma
        self.beta -= learning_rate * dCdbeta
        return dCdx
    