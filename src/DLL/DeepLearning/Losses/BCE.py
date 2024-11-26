import torch

from .BaseLoss import BaseLoss


class bce(BaseLoss):
    """
    The binary cross entropy loss. Used in binary classification. Identical to categorical cross entropy with 2 classes.

    Args:
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
    """
    def __init__(self, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError('reduction must be in ["mean", "sum"].')

        self.reduction = reduction

    def loss(self, prediction, true_output):
        """
        Calculates the binary categorical cross entropy with the equations:

        .. math::
        
            \\begin{align*}
                l_i &= y_i\\cdot\\text{ln}(f(x_i)) + (1 - y_i)\\cdot\\text{ln}(1 - f(x_i)),\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,
            \\end{align*}

        where :math:`f(x_i)` is the predicted value and :math:`y_i` is the true value.

        Args:
            prediction (torch.Tensor): A tensor of predicted values in range [0, 1]. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values labeled with 0 or 1. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")

        if self.reduction == "mean":
            return -torch.mean(true_output * torch.log(prediction + 1e-10) + (1 - true_output) * torch.log(1 - prediction + 1e-10))
        return -torch.sum(true_output * torch.log(prediction + 1e-10) + (1 - true_output) * torch.log(1 - prediction + 1e-10))

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the binary categorical cross entropy.

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

        if self.reduction == "mean":
            return (prediction - true_output) / ((prediction * (1 - prediction) + 1e-10) * prediction.shape[0])
        return (prediction - true_output) / (prediction * (1 - prediction) + 1e-10)
    
    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the binary categorical cross entropy.

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
        
        first_term = 1 / ((1 - prediction) * prediction + 1e-10)
        second_term = (true_output - prediction) / ((1 - prediction) * prediction ** 2 + 1e-10)
        third_term = (prediction - true_output) / ((1 - prediction) ** 2 * prediction + 1e-10)
        hess = first_term + second_term + third_term

        if self.reduction == "mean":
            return hess / prediction.shape[0]
        return hess
        
