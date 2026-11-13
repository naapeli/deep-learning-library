import torch

from ._BaseLoss import BaseLoss, _check_reduction, _check_bool, _check_binary_labels, _check_smoothing, _margin_loss


class Hinge(BaseLoss):
    """
    The hinge loss. Used in maximum margin classification, for instance in support vector machines. The predictions are raw scores (not probabilities), and a sample only contributes to the loss if it is misclassified or inside the margin.

    Args:
        squared (bool, optional): If True, the squared hinge loss is used, which is differentiable and penalises large violations more. Defaults to False.
        categorical (bool, optional): If False, every element is an independent binary problem with labels 0 and 1, which are mapped to -1 and 1. If True, the last dimension is the class dimension and the labels are one-hot encoded, and the multi-class hinge loss of Weston and Watkins is used. Defaults to False.
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
        smoothing (int | float, optional): If positive, the corner of the hinge is replaced by a quadratic function on the margins in [0, smoothing], which makes the loss differentiable with a nonzero hessian. As the smoothing goes to zero, the loss approaches the hinge loss. Can only be used with squared=False. Defaults to 0.
    """
    def __init__(self, squared=False, categorical=False, reduction="mean", smoothing=0):
        _check_bool(squared, "squared")
        _check_bool(categorical, "categorical")
        _check_reduction(reduction)
        _check_smoothing(smoothing, squared)

        self.squared = squared
        self.categorical = categorical
        self.reduction = reduction
        self.smoothing = smoothing

    def _check(self, prediction, true_output):
        self._check_inputs(prediction, true_output)
        _check_binary_labels(true_output)
        if self.categorical and torch.any(true_output.sum(dim=-1) != 1):
            raise ValueError("With categorical=True, the labels must be one-hot encoded over the last dimension.")

    def _n_samples(self, prediction):
        return prediction.numel() // prediction.shape[-1] if self.categorical else prediction.numel()

    def _margins(self, prediction, true_output):
        if self.categorical:
            # 1 + f_j - f_y for every wrong class j, 0 for the true class
            true_score = (prediction * true_output).sum(dim=-1, keepdim=True)
            return torch.clamp(1 + prediction - true_score, min=0) * (1 - true_output)
        sign = 2 * true_output - 1
        return torch.clamp(1 - sign * prediction, min=0)

    def loss(self, prediction, true_output):
        """
        Calculates the hinge loss with the equations:

        .. math::

                l_i &= \\max(0, 1 - t_i f(x_i))^p,\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`f(x_i)` is the predicted score, :math:`t_i = 2y_i - 1\\in\\{-1, 1\\}` is the true label and :math:`p` is 2 if squared=True and 1 otherwise. With categorical=True,

        .. math::

                l_i = \\sum_{j\\neq y_i} \\max(0, 1 + f_j(x_i) - f_{y_i}(x_i))^p,

        where :math:`y_i` is the index of the true class.

        Args:
            prediction (torch.Tensor): A tensor of predicted scores. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true labels with values 0 or 1. If categorical=True, must be one-hot encoded over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check(prediction, true_output)

        loss, _, _ = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        return self._reduce(loss, self._n_samples(prediction))

    def gradient(self, prediction, true_output):
        """
        Calculates the (sub)gradient of the hinge loss.

        Args:
            prediction (torch.Tensor): A tensor of predicted scores. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true labels with values 0 or 1. If categorical=True, must be one-hot encoded over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check(prediction, true_output)

        # derivative of the loss with respect to each margin
        _, dloss_dmargin, _ = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        if self.categorical:
            # the margin of class j grows with f_j and shrinks with f_y
            grad = dloss_dmargin - true_output * dloss_dmargin.sum(dim=-1, keepdim=True)
        else:
            grad = -(2 * true_output - 1) * dloss_dmargin
        return self._scale(grad, self._n_samples(prediction))

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the hinge loss. The hinge loss is piecewise linear, so its hessian is zero. The squared hinge loss has the hessian 2 for every active margin, and the smoothed hinge loss 1 / smoothing for every margin in (0, smoothing).

        Args:
            prediction (torch.Tensor): A tensor of predicted scores. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true labels with values 0 or 1. If categorical=True, must be one-hot encoded over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        self._check(prediction, true_output)

        # the margins are linear in the predictions, so only the second derivative with respect to the margins remains
        _, _, d2loss_dmargin2 = _margin_loss(self._margins(prediction, true_output), self.squared, self.smoothing)
        if self.categorical:
            hess = d2loss_dmargin2 + true_output * d2loss_dmargin2.sum(dim=-1, keepdim=True)
        else:
            hess = d2loss_dmargin2
        return self._scale(hess, self._n_samples(prediction))
