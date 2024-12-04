import torch

from ._BaseLoss import BaseLoss


class Exponential(BaseLoss):
    """
    The exponential loss. Used for binary classification.

    Args:
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
    """
    def __init__(self, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError('reduction must be in ["mean", "sum"].')

        self.reduction = reduction

    def loss(self, prediction, true_output):
        """
        Calculates the exponential loss with the equations:

        .. math::

            \\begin{align*}
                l_i &= y_i\\cdot e^{-f(x_i)} + (1 - y_i)\\cdot e^{f(x_i)},\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,
            \\end{align*}

        where :math:`f(x_i)` is the predicted value and :math:`y_i` is the true value.

        Args:
            prediction (torch.Tensor): A tensor of predicted values as a probability distribution. Must be the same shape as the true_output.
            true_output (torch.Tensor): A one-hot encoded tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")
        if set(torch.unique(true_output).numpy()) != {0, 1}:
            raise ValueError("The classes must be labelled 0 and 1.")

        prediction = 2 * prediction - 1
        true_output = 2 * true_output - 1
        loss = torch.exp(-prediction * true_output)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the exponential loss.

        Args:
            prediction (torch.Tensor): A tensor of predicted values in range [0, 1]. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values labeled with 0 or 1. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")
        if set(torch.unique(true_output).numpy()) != {0, 1}:
            raise ValueError("The classes must be labelled 0 and 1.")
        
        prediction = 2 * prediction - 1
        true_output = 2 * true_output - 1
        grad = -2 * true_output * torch.exp(-prediction * true_output)
        if self.reduction == "mean":
            return grad / prediction.shape[0]
        return grad
    
    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the exponential loss.

        Args:
            prediction (torch.Tensor): A tensor of predicted values in range [0, 1]. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values labeled with 0 or 1. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")
        if set(torch.unique(true_output).numpy()) != {0, 1}:
            raise ValueError("The classes must be labelled 0 and 1.")
        
        prediction = 2 * prediction - 1
        true_output = 2 * true_output - 1
        grad = 4 * true_output ** 2 * torch.exp(-prediction * true_output)
        if self.reduction == "mean":
            return grad / prediction.shape[0]
        return grad
