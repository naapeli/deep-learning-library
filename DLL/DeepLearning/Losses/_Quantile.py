import torch

from ._BaseLoss import BaseLoss, _check_reduction


class Quantile(BaseLoss):
    """
    The quantile (pinball) loss. Used for quantile regression, i.e. predicting a chosen quantile of the target distribution instead of its mean. With quantile=0.5, the loss is half of the absolute error and the model predicts the median. Training models for a low and a high quantile gives a prediction interval.

    Args:
        quantile (float, optional): The quantile :math:`\\tau` that is predicted. Must be in range (0, 1). Defaults to 0.5.
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
    """
    def __init__(self, quantile=0.5, reduction="mean"):
        if not isinstance(quantile, float) or quantile <= 0 or quantile >= 1:
            raise ValueError("quantile must be a float in range (0, 1).")
        _check_reduction(reduction)

        self.quantile = quantile
        self.reduction = reduction

    def loss(self, prediction, true_output):
        """
        Calculates the quantile loss with the equations:

        .. math::

                l_i &= \\max(\\tau(y_i - f(x_i)), (\\tau - 1)(y_i - f(x_i))),\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`f(x_i)` is the predicted value, :math:`y_i` is the true value and :math:`\\tau` is the quantile.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check_inputs(prediction, true_output)

        error = true_output - prediction
        loss = torch.maximum(self.quantile * error, (self.quantile - 1) * error)
        return self._reduce(loss, prediction.numel())

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the quantile loss. The gradient is :math:`-\\tau` for underestimates, :math:`1 - \\tau` for overestimates and 0 for exact predictions.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check_inputs(prediction, true_output)

        error = true_output - prediction
        grad = torch.where(error > 0, -self.quantile, 1 - self.quantile)
        grad = torch.where(error == 0, 0.0, grad).to(prediction.dtype)
        return self._scale(grad, prediction.numel())

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the quantile loss. The loss is piecewise linear, so the hessian is zero.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of zeros with the same shape as the inputs.
        """
        self._check_inputs(prediction, true_output)

        return torch.zeros_like(prediction)
