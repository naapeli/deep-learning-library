import torch

from ._Activation import Activation


class LeakyReLU(Activation):
    """
    The leaky rectified linear unit activation function.

    .. math::

        f(x) = \\begin{cases} x & \\text{if } x > 0,\\\\ \\alpha x & \\text{otherwise}. \\end{cases}

    Args:
        negative_slope (float, optional): The slope :math:`\\alpha` for negative inputs. Must be a non-negative real number. Defaults to 0.01.
    """
    def __init__(self, negative_slope=0.01, **kwargs):
        if not isinstance(negative_slope, int | float) or negative_slope < 0:
            raise ValueError("negative_slope must be a non-negative real number.")
        super().__init__(**kwargs)
        self.name = "LeakyReLU"
        self.negative_slope = negative_slope

    def forward(self, input, **kwargs):
        """
        Calculates the leaky ReLU function for every element of the input matrix.

        Args:
            input (torch.Tensor of shape (batch_size, ...)): The input to the layer. Must be a torch.Tensor of any shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")

        self.input = input
        return torch.where(input > 0, input, self.negative_slope * input)

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

        return dCdy * torch.where(self.input > 0, torch.ones_like(self.input), torch.full_like(self.input, self.negative_slope))
