import torch

from .BaseLayer import BaseLayer


class Reshape(BaseLayer):
    """
    The reshape layer.

    Args:
        output_shape (int): The output_shape of the model not containing the batch_size dimension. Must be a positive integer or a tuple.
    """
    def __init__(self, output_shape, **kwargs):
        output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape
        if not isinstance(output_shape, tuple | list):
            raise TypeError("output_shape must be a tuple or an integer.")

        super().__init__(output_shape, **kwargs)
        self.name = "Reshape"
    
    def forward(self, input, **kwargs):
        """
        Reshapes the input into the output_shape.

        Args:
            input (torch.Tensor of shape (n_samples, *layer.input_shape)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.

        Returns:
            torch.Tensor of shape (n_samples, *layer.output_shape): The output tensor after reshaping the input tensor.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")

        self.input = input
        return input.reshape(input.shape[0], *self.output_shape)
    
    def backward(self, dCdy, **kwargs):
        """
        Reshapes the gradient to the original shape.

        Args:
            dCdy (torch.Tensor of shape (n_samples, *layer.output_shape): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, *layer.input_shape): The reshaped gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        return dCdy.reshape(*self.input.shape)
