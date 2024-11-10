import torch
import numpy as np

from .BaseRegularisation import BaseRegularisation


class BatchNorm(BaseRegularisation):
    """
    The batch normalisation layer for neural networks.

    Args:
        patience (float, optional): The number deciding how fast the mean and variance in training. Must be strictly between 0 and 1. Defaults to 0.9.
    """
    def __init__(self, patience=0.9, **kwargs):
        if patience <= 0 or 1 <= patience:
            raise ValueError("patience must be strictly between 0 and 1.")
        
        super().__init__(**kwargs)
        self.patience = patience
        self.epsilon = 1e-6
        self.name = "Batch normalisation"
    
    def initialise_layer(self, **kwargs):
        """
        :meta private:
        """

        super().initialise_layer(**kwargs)

        self.gamma = torch.ones(self.output_shape, dtype=self.data_type, device=self.device)
        self.beta = torch.zeros(self.output_shape, dtype=self.data_type, device=self.device)
        self.running_var = torch.ones(self.output_shape, dtype=self.data_type, device=self.device)
        self.running_mean = torch.zeros(self.output_shape, dtype=self.data_type, device=self.device)
        self.nparams = 2 * np.prod(self.output_shape)

    def forward(self, input, training=False, **kwargs):
        """
        Normalises the input to have zero mean and one variance with the following equation:
        
        .. math::
            y = \\gamma\\frac{x - \\mathbb{E}[x]}{\\sqrt{\\text{var}(x) + \\epsilon}} + \\beta,
        
        where :math:`x` is the input, :math:`\\mathbb{E}[x]` is the expected value or the mean accross the batch dimension, :math:`\\text{var}(x)` is the variance accross the variance accross the batch dimension, :math:`\\epsilon` is a small constant and :math:`\\gamma` and :math:`\\beta` are trainable parameters.

        Args:
            input (torch.Tensor of shape (batch_size, channels, ...)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The output tensor after the normalisation with the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")
        if input.shape[0] <= 1:
            raise ValueError("The batch size must be atleast 2.")
        
        self.input = input
        if training:
            mean = torch.mean(input, axis=0)
            variance = torch.var(input, axis=0, unbiased=True)
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
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (batch_size, channels, ...): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        batch_size = self.output.shape[0]
        dCdx_norm = dCdy * self.gamma
        dCdgamma = (dCdy * self.x_norm).mean(axis=0)
        dCdbeta = dCdy.mean(axis=0)
        dCdvar = (dCdx_norm * self.x_centered * -self.std**(-3) / 2).sum(axis=0)
        dCdmean = -((dCdx_norm / self.std).sum(axis=0) + dCdvar * (2 / batch_size) * self.x_centered.sum(axis=0))
        dCdx = dCdx_norm / self.std + dCdvar * 2 * self.x_centered / (batch_size - 1) + dCdmean / batch_size

        self.gamma.grad = dCdgamma
        self.beta.grad = dCdbeta
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (self.gamma, self.beta)
