import torch

from ._BaseLoss import BaseLoss, _check_reduction, _check_bool, _check_smoothing, _margin_loss


class EpsilonInsensitive(BaseLoss):
    """
    The epsilon-insensitive loss. Used in support vector regression. Errors smaller than epsilon are not penalised, so the model fits a tube of width 2 * epsilon around the data, and only the samples outside the tube affect the solution.

    Args:
        epsilon (int | float, optional): The half width of the tube of no penalty. Must be a non-negative real number. Defaults to 0.1.
        squared (bool, optional): If True, the squared epsilon-insensitive loss is used, which is differentiable and penalises large errors more. Defaults to False.
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
        smoothing (int | float, optional): If positive, the corners of the loss at the edges of the tube are replaced by quadratic functions on the margins in [0, smoothing], which makes the loss differentiable with a nonzero hessian. As the smoothing goes to zero, the loss approaches the epsilon-insensitive loss. Can only be used with squared=False. Defaults to 0.
    """
    def __init__(self, epsilon=0.1, squared=False, reduction="mean", smoothing=0):
        if not isinstance(epsilon, int | float) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative real number.")
        _check_bool(squared, "squared")
        _check_reduction(reduction)
        _check_smoothing(smoothing, squared)

        self.smoothing = smoothing
        self.epsilon = epsilon
        self.squared = squared
        self.reduction = reduction

    def _margins(self, prediction, true_output):
        return torch.clamp(torch.abs(prediction - true_output) - self.epsilon, min=0)

    def loss(self, prediction, true_output):
        """
        Calculates the epsilon-insensitive loss with the equations:

        .. math::

                l_i &= \\max(0, |y_i - f(x_i)| - \\varepsilon)^p,\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`f(x_i)` is the predicted value, :math:`y_i` is the true value and :math:`p` is 2 if squared=True and 1 otherwise.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check_inputs(prediction, true_output)

        loss, _, _ = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        return self._reduce(loss, prediction.numel())

    def gradient(self, prediction, true_output):
        """
        Calculates the (sub)gradient of the epsilon-insensitive loss.

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check_inputs(prediction, true_output)

        _, dloss_dmargin, _ = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        return self._scale(torch.sign(prediction - true_output) * dloss_dmargin, prediction.numel())

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the epsilon-insensitive loss. The loss is piecewise linear, so its hessian is zero. The squared loss has the hessian 2 outside the tube, and the smoothed loss 1 / smoothing for the margins in (0, smoothing).

        Args:
            prediction (torch.Tensor): A tensor of predicted values. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        self._check_inputs(prediction, true_output)

        _, _, d2loss_dmargin2 = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        return self._scale(d2loss_dmargin2, prediction.numel())
