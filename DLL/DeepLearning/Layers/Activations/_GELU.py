import math

import torch

from ._Activation import Activation


class GELU(Activation):
    """
    The Gaussian error linear unit (GELU) activation function.

    .. math::

        f(x) = x\\Phi(x) = \\frac{x}{2}\\left(1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right),

    where :math:`\\Phi` is the cumulative distribution function of the standard normal distribution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GELU"

    def forward(self, input, **kwargs):
        """
        Calculates the GELU function for every element of the input matrix.

        Args:
            input (torch.Tensor of shape (batch_size, ...)): The input to the layer. Must be a torch.Tensor of any shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")

        self.input = input
        return 0.5 * input * (1 + torch.erf(input / math.sqrt(2)))

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

        cdf = 0.5 * (1 + torch.erf(self.input / math.sqrt(2)))
        pdf = torch.exp(-0.5 * self.input ** 2) / math.sqrt(2 * math.pi)
        return dCdy * (cdf + self.input * pdf)
