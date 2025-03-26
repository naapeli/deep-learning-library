import torch

from ._Activation import Activation


class Sigmoid(Activation):
    """
    The sigmoid activation function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Sigmoid"

    def forward(self, input, **kwargs):
        """
        Calculates the following function for every element of the input matrix:

        .. math::
        
            \\sigma(x) = \\frac{1}{1 + e^{-x}}.

        Args:
            input (torch.Tensor of shape (batch_size, ...)): The input to the layer. Must be a torch.Tensor of any shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        
        self.input = input
        self.output = 1 / (1 + torch.exp(-self.input))
        return self.output
    
    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (batch_size, ...): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape != self.output.shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        dCdx = (self.output * (1 - self.output)) * dCdy
        return dCdx
