import torch

from .BaseLoss import BaseLoss


class Huber(BaseLoss):
    """
    The huber loss. Used for regression. Is a combination of squared error and absolute error.

    Args:
        delta (int | float, optional): The radius around the true value that uses the squared error. If the difference is larger than delta, the absolute error is used. Must be a positive real number. Defaults to 1.
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum".
    """
    def __init__(self, delta=1.0, reduction="mean"):
        if not isinstance(delta, int | float) or delta <= 0:
            raise ValueError("delta must be a positive real number.")
        if reduction not in ["mean", "sum"]:
            raise ValueError('reduction must be in ["mean", "sum"].')
        
        self.delta = delta
        self.reduction = reduction

    def loss(self, prediction, true_output):
        """
        Calculates the huber loss with the equations:

        .. math::
        
            l_i =
            \\begin{cases}
                \\frac{1}{2}(y_i - f(x_i))^2 & \\text{if \\abs{y_i - f(x_i)} <= \\delta},\\\\
                \\delta\\abs{y_i - f(x_i)} - \\frac{1}{2}\\delta^2 & \\text{otherwise},
            \end{cases}\\\\
            L_{sum} = \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

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
        
        error = prediction - true_output
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        if self.reduction == "mean":
            return torch.where(abs_error <= self.delta, quadratic, linear).mean()
        return torch.where(abs_error <= self.delta, quadratic, linear).sum()

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
        
        error = prediction - true_output
        abs_error = torch.abs(error)
        quadratic_grad = error
        linear_grad = self.delta * torch.sign(error)
        if self.reduction == "mean":
            return torch.where(abs_error <= self.delta, quadratic_grad, linear_grad) / prediction.shape[0]
        return torch.where(abs_error <= self.delta, quadratic_grad, linear_grad)
