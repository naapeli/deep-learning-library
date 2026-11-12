import torch

from ._BaseLoss import BaseLoss, _check_reduction, _check_bool, _check_soft_labels, _log_sigmoid


class BCE(BaseLoss):
    """
    The binary cross entropy loss. Used in binary and multi-label classification. Identical to categorical cross entropy with 2 classes. Every element of the inputs is treated as an independent binary problem.

    Args:
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
        from_logits (bool, optional): If True, the predictions are logits (real numbers), which are mapped to probabilities with the sigmoid function inside the loss. This is numerically more stable than applying a sigmoid activation to the last layer, and the gradient with respect to the logits simplifies to :math:`\\sigma(z) - y`. A model trained with this loss returns logits from ``predict``. Defaults to False.

    Note:
        The labels can be soft, i.e. any value in [0, 1] instead of only 0 or 1. This is used for instance in label smoothing or knowledge distillation.
    """
    def __init__(self, reduction="mean", from_logits=False):
        _check_reduction(reduction)
        _check_bool(from_logits, "from_logits")

        self.reduction = reduction
        self.from_logits = from_logits

    def _check(self, prediction, true_output):
        self._check_inputs(prediction, true_output)
        _check_soft_labels(true_output)

    def loss(self, prediction, true_output):
        """
        Calculates the binary cross entropy with the equations:

        .. math::

                l_i &= -y_i\\cdot\\text{ln}(p_i) - (1 - y_i)\\cdot\\text{ln}(1 - p_i),\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`p_i = f(x_i)` is the predicted probability, or :math:`p_i = \\sigma(f(x_i))` if from_logits=True, and :math:`y_i` is the true value.

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities in range [0, 1], or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. Usually 0 or 1. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check(prediction, true_output)

        if self.from_logits:
            # -y * log(sigmoid(z)) - (1 - y) * log(1 - sigmoid(z)), where log(1 - sigmoid(z)) = log(sigmoid(z)) - z
            loss = (1 - true_output) * prediction - _log_sigmoid(prediction)
        else:
            loss = -(true_output * torch.log(prediction + 1e-10) + (1 - true_output) * torch.log(1 - prediction + 1e-10))
        return self._reduce(loss, prediction.numel())

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the binary cross entropy. With from_logits=True, the gradient with respect to the logits is :math:`\\sigma(z) - y`.

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities in range [0, 1], or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check(prediction, true_output)

        if self.from_logits:
            grad = torch.sigmoid(prediction) - true_output
        else:
            grad = (1 - true_output) / (1 - prediction + 1e-10) - true_output / (prediction + 1e-10)
        return self._scale(grad, prediction.numel())

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the binary cross entropy. With from_logits=True, the diagonal with respect to the logits is :math:`\\sigma(z)(1 - \\sigma(z))`.

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities in range [0, 1], or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        self._check(prediction, true_output)

        if self.from_logits:
            probability = torch.sigmoid(prediction)
            hess = probability * (1 - probability)
        else:
            hess = true_output / (prediction + 1e-10) ** 2 + (1 - true_output) / (1 - prediction + 1e-10) ** 2
        return self._scale(hess, prediction.numel())

    def transform_prediction(self, prediction):
        """
        Returns the predicted probabilities. Applies the sigmoid if from_logits=True, otherwise returns the predictions unchanged.

        Args:
            prediction (torch.Tensor): The raw output of a model.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        return torch.sigmoid(prediction) if self.from_logits else prediction
