import torch

from ._Activation import Activation


class ELU(Activation):
    """
    The exponential linear unit (ELU) activation function.

    .. math::

        f(x) = \\begin{cases} x & \\text{if } x > 0,\\\\ \\alpha(e^x - 1) & \\text{otherwise}. \\end{cases}

    Args:
        alpha (float, optional): The scale :math:`\\alpha` of the negative part. Must be a positive real number. Defaults to 1.0.
    """
    def __init__(self, alpha=1.0, **kwargs):
        if not isinstance(alpha, int | float) or alpha <= 0:
            raise ValueError("alpha must be a positive real number.")
        super().__init__(**kwargs)
        self.name = "ELU"
        self.alpha = alpha

    def forward(self, input, **kwargs):
        """
        Calculates the ELU function for every element of the input matrix.

        Args:
            input (torch.Tensor of shape (batch_size, ...)): The input to the layer. Must be a torch.Tensor of any shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")

        self.input = input
        return torch.where(input > 0, input, self.alpha * torch.expm1(input.clamp(max=0)))

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

        return dCdy * torch.where(self.input > 0, torch.ones_like(self.input), self.alpha * torch.exp(self.input.clamp(max=0)))
