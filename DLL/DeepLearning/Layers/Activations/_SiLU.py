import torch

from ._Activation import Activation


class SiLU(Activation):
    """
    The sigmoid linear unit (SiLU), also known as the swish activation function.

    .. math::

        f(x) = x\\sigma(x) = \\frac{x}{1 + e^{-x}}
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SiLU"

    def forward(self, input, **kwargs):
        """
        Calculates the SiLU function for every element of the input matrix.

        Args:
            input (torch.Tensor of shape (batch_size, ...)): The input to the layer. Must be a torch.Tensor of any shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")

        self.input = input
        return input * torch.sigmoid(input)

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
        if dCdy.shape != self.input.shape:
            raise ValueError(f"dCdy is not the same shape as the input ({dCdy.shape[1:], self.input.shape[1:]}).")

        sigmoid = torch.sigmoid(self.input)
        return dCdy * (sigmoid + self.input * sigmoid * (1 - sigmoid))
