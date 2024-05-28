import torch
from ..Activations.Activation import Activation


class BatchNorm1d(Activation):
    def __init__(self, output_shape=None, patience=0.9, **kwargs):
        super().__init__(output_shape, **kwargs)
        assert 0 < patience and patience < 1, "Patience must be strictly between 0 and 1"
        self.patience = patience
        self.epsilon = 1e-6
        self.name = "Batch normalisation"
        if output_shape is not None: self.set_output_shape(output_shape)
    
    def set_output_shape(self, output_shape):
        self.output_shape = output_shape
        self.input_shape = output_shape
        self.gamma = torch.ones(self.output_shape[-1], dtype=self.data_type, device=self.device)
        self.beta = torch.zeros(self.output_shape[-1], dtype=self.data_type, device=self.device)
        self.running_var = torch.ones(self.output_shape[-1], dtype=self.data_type, device=self.device)
        self.running_mean = torch.zeros(self.output_shape[-1], dtype=self.data_type, device=self.device)
        self.nparams = 2 * self.output_shape[-1]

    def forward(self, input, training=False, **kwargs):
        # MODIFY TO TAKE arbitrarily shaped inputs
        self.input = input
        if training:
            mean = torch.mean(input, axis=0)
            variance = torch.var(input, axis=0, unbiased=True) if self.input.shape[0] > 1 else torch.zeros(self.output_shape[-1], dtype=self.data_type, device=self.device)
            self.std = torch.sqrt(variance + self.epsilon)
            self.running_mean = self.patience * self.running_mean + (1 - self.patience) * mean
            self.running_var = self.patience * self.running_var + (1 - self.patience) * variance
            self.x_centered = (self.input - mean)
            self.x_norm = self.x_centered / self.std
            self.output = self.gamma * self.x_norm + self.beta
        else:
            self.output = self.gamma * ((self.input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)) + self.beta
        return self.output
    
    def backward(self, dCdy, **kwargs):
        batch_size = self.output.shape[0]
        dCdx_norm = dCdy * self.gamma
        dCdgamma = (dCdy * self.x_norm).sum(axis=0)
        dCdbeta = dCdy.sum(axis=0)
        dCdvar = (dCdx_norm * self.x_centered * -self.std**(-3) / 2).sum(axis=0)
        dCdmean = -((dCdx_norm / self.std).sum(axis=0) + dCdvar * (2 / batch_size) * self.x_centered.sum(axis=0))
        dCdx = dCdx_norm / self.std + (dCdvar * 2 * self.x_centered + dCdmean) / batch_size

        self.gamma.grad = dCdgamma
        self.beta.grad = dCdbeta
        return dCdx
    
    def summary(self):
        return f"{self.name} - Output: ({self.output_shape}) - Parameters: {self.nparams}"
    
    def get_parameters(self):
        return (self.gamma, self.beta)
    