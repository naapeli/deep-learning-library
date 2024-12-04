import torch

from ._BaseRegularisation import BaseRegularisation


class GroupNorm(BaseRegularisation):
    """
    The group normalisation layer for neural networks. Computes the group norm of a batch along axis=1

    Args:
        num_groups (int, optional): The number of groups used in the normalisation. Must be a positive integer. Defaults to 32. The number of channels must be evenly divisible by num_groups. If is set to 1, is identical to layer normalisation and if batch_size, is identical to the instance normalisation.
    """
    def __init__(self, num_groups=32, **kwargs):
        if num_groups is not None and (not isinstance(num_groups, int) or num_groups <= 0):
            raise ValueError("num_groups must be a positive integer.")

        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.epsilon = 1e-6
        self.name = "Group normalisation"
    
    def initialise_layer(self, **kwargs):
        """
        :meta private:
        """

        super().initialise_layer(**kwargs)

        if self.output_shape[0] % self.num_groups != 0:
            raise ValueError("The number of channels must be evenly divisible by num_groups.")
        
        self.gamma = torch.ones(self.output_shape, device=self.device, dtype=self.data_type)
        self.beta = torch.zeros(self.output_shape, device=self.device, dtype=self.data_type)
        self.nparams = 2 * self.output_shape[0]

    def forward(self, input, **kwargs):
        """
        Normalises the input to have zero mean and one variance accross self.num_groups groups accross the channel dimension with the following equation:
        
        .. math::
            y = \\gamma\\frac{x - \\mathbb{E}[x]}{\\sqrt{\\text{var}(x) + \\epsilon}} + \\beta,
        
        where :math:`x` is the input, :math:`\\mathbb{E}[x]` is the expected value or the mean accross each group, :math:`\\text{var}(x)` is the variance accross the variance accross each group, :math:`\\epsilon` is a small constant and :math:`\\gamma` and :math:`\\beta` are trainable parameters.

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

        elements_per_group = input.shape[1] // self.num_groups
        self.input = input
        batch_size = input.shape[0]
        self.input = input
        self.input_reshaped = self.input.view(batch_size, self.num_groups, elements_per_group, *input.shape[2:])
        mean = 1.0 / elements_per_group * self.input_reshaped.sum(2, keepdim=True)

        self.x_centered = self.input_reshaped - mean
        self.x_centered_squared = self.x_centered ** 2
        self.var = 1.0 / elements_per_group * self.x_centered_squared.sum(2, keepdim=True)  # biased variance

        self.inv_std = (self.var + self.epsilon) ** -0.5
        self.x_norm = self.x_centered * self.inv_std
        self.x_reshaped = self.x_norm.view(self.input.shape)
        self.output = self.x_reshaped * self.gamma + self.beta
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
        
        batch_size = dCdy.shape[0]
        elements_per_group = self.output.shape[1] // self.num_groups
        dCdx_reshaped = dCdy * self.gamma
        dCdgamma = (dCdy * self.x_reshaped).mean(axis=0)
        dCdbeta = dCdy.mean(axis=0)
        self.gamma.grad = dCdgamma
        self.beta.grad = dCdbeta

        dCdx_norm = dCdx_reshaped.view(batch_size, self.num_groups, elements_per_group, *self.output.shape[2:])
        dCdx_centered = dCdx_norm * self.inv_std
        dCdinv_std = (dCdx_norm * self.x_centered).sum(2, keepdim=True)
        dCdvar = -0.5 * ((self.var + self.epsilon) ** -1.5) * dCdinv_std
        dCdx_centered_squared = 1.0 / elements_per_group * torch.ones_like(self.x_centered_squared, device=self.device, dtype=self.data_type) * dCdvar
        dCdx_centered += 2 * self.x_centered * dCdx_centered_squared
        dCdinput_reshaped = dCdx_centered.clone()
        dCdmean = -(dCdx_centered).sum(2, keepdim=True)
        dCdinput_reshaped += 1.0 / elements_per_group * torch.ones_like(self.input_reshaped, device=self.device, dtype=self.data_type) * dCdmean
        dCdx = dCdinput_reshaped.view(self.output.shape)
        return dCdx.view(self.output.shape)

    def get_parameters(self):
        """
        :meta private:
        """
        return (self.gamma, self.beta)
