import torch
from Layers.Activations.Activation import Activation


class GroupNorm1d(Activation):
    def __init__(self, output_size=1, num_groups=32, **kwargs):
        # assert output_size % num_groups == 0, "output_size must be divisible by the number of groups"
        super().__init__(output_size)
        self.gamma = torch.ones(self.output_size)
        self.beta = torch.zeros(self.output_size)
        self.num_groups = num_groups
        self.epsilon = 1e-6
        self.x_centered = None
        self.x_norm = None
        self.x_reshaped = None
        self.name = "Group normalisation"

    def forward(self, input, **kwargs):
        batch_size = input.shape[0]
        elements_per_group = self.output_size // self.num_groups
        self.input = input
        self.input_reshaped = self.input.view(batch_size, self.num_groups, elements_per_group)
        mean = 1.0 / elements_per_group * self.input_reshaped.sum(2, keepdim=True)
        self.x_centered = self.input_reshaped - mean
        self.x_centered_squared = self.x_centered ** 2
        # unbiased variance
        self.var = 1.0 / (elements_per_group - 1) * self.x_centered_squared.sum(2, keepdim=True) if elements_per_group > 1 else torch.zeros(size=(batch_size, self.num_groups, 1), dtype=self.data_type, device=self.device)
        self.inv_std = (self.var + self.epsilon) ** -0.5
        self.x_norm = self.x_centered * self.inv_std
        self.x_reshaped = self.x_norm.view(self.input.shape)
        self.output = self.x_reshaped * self.gamma + self.beta
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001, **kwargs):
        batch_size = self.output.shape[0]
        elements_per_group = self.output_size // self.num_groups
        dCdx_reshaped = dCdy * self.gamma
        dCdgamma = (dCdy * self.x_reshaped).sum(axis=0)
        dCdbeta = dCdy.sum(axis=0)
        self.gamma -= learning_rate * dCdgamma
        self.beta -= learning_rate * dCdbeta

        dCdx_norm = dCdx_reshaped.view(batch_size, self.num_groups, elements_per_group)
        dCdx_centered = dCdx_norm * self.inv_std
        dCdinv_std = (dCdx_norm * self.x_centered).sum(2, keepdim=True)
        dCdvar = -0.5 * ((self.var + self.epsilon) ** -1.5) * dCdinv_std
        dCdx_centered_squared = 1.0 / (elements_per_group - 1) * torch.ones_like(self.x_centered_squared) * dCdvar
        dCdx_centered += 2 * self.x_centered * dCdx_centered_squared
        dCdinput_reshaped = dCdx_centered.clone()
        dCdmean = -(dCdx_centered).sum(2, keepdim=True)
        dCdinput_reshaped += 1.0 / elements_per_group * (torch.ones_like(self.input_reshaped) * dCdmean)
        dCdx = dCdinput_reshaped.view(self.output.shape)
        return dCdx.view(self.output.shape)
    
    def summary(self):
        return f"{self.name} - Output: ({self.output_size}) - Parameters: {self.nparams}"
    