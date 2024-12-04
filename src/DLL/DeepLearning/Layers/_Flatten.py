import numpy as np
import torch

from ._BaseLayer import BaseLayer


class Flatten(BaseLayer):
    """
    The flattening layer.
    """
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
        self.name = "Flatten"
    
    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if not isinstance(input_shape, tuple | list):
            raise ValueError("input_shape must be a tuple.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        self.output_shape = (np.prod(input_shape),)
        super().initialise_layer(input_shape, data_type, device)
    
    def forward(self, input, **kwargs):
        """
        Flattens the input tensor into a 2 dimensional tensor.

        Args:
            input (torch.Tensor of shape (n_samples, ...)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            
        Returns:
            torch.Tensor of shape (n_samples, product_of_other_dimensions): The output tensor after flattening the input tensor.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")

        self.input = input
        return input.reshape(input.shape[0], -1)
    
    def backward(self, dCdy, **kwargs):
        """
        Reshapes the gradient to the original shape.

        Args:
            dCdy (torch.Tensor of shape (n_samples, product_of_other_dimensions): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, *layer.input_shape): The reshaped gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        return dCdy.reshape(*self.input.shape)
