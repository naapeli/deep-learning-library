import torch

from .BaseLoss import BaseLoss


class mse(BaseLoss):
    """
    The squared error loss. Used for regression.

    Args:
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum".
    """
    def __init__(self, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError('reduction must be in ["mean", "sum"].')
        
        self.reduction = reduction

    def loss(self, prediction, true_output):
        """
        Calculates the absolute error loss with the equations:

        .. math::

            \\begin{align*}
                l_i &= (y_i - f(x_i))^2,\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,
            \\end{align*}
        
        where :math:`f(x_i)` is the predicted value and :math:`y_i` is the true value.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")
        
        if self.reduction == "mean":
            return torch.mean((prediction - true_output) ** 2)
        return torch.sum((prediction - true_output) ** 2)

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the huber loss.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")
        
        if self.reduction == "mean":
            return 2 * (prediction - true_output) / prediction.shape[0]
        return 2 * (prediction - true_output)
