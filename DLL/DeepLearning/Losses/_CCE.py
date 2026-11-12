import torch

from ._BaseLoss import BaseLoss, _check_reduction, _check_bool, _log_softmax


class CCE(BaseLoss):
    """
    The categorical cross entropy loss. Used in multi-class classification. The last dimension of the inputs is the class dimension. If the tensors have more than 2 dimensions, for instance (n_samples, seq_len, n_classes), every position is treated as its own sample.

    Args:
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".
        from_logits (bool, optional): If True, the predictions are logits (real numbers), which are mapped to probabilities with the softmax function over the last dimension inside the loss. This is numerically more stable than applying a softmax activation to the last layer, and the gradient with respect to the logits simplifies to :math:`\\text{softmax}(z) - y`. A model trained with this loss returns logits from ``predict``. Defaults to False.

    Note:
        The labels can be soft, i.e. any probability distribution over the classes instead of a one-hot encoded vector. This is used for instance in label smoothing or knowledge distillation.
    """
    def __init__(self, reduction="mean", from_logits=False):
        _check_reduction(reduction)
        _check_bool(from_logits, "from_logits")

        self.reduction = reduction
        self.from_logits = from_logits

    def _check(self, prediction, true_output):
        self._check_inputs(prediction, true_output)
        if torch.any(true_output < 0):
            raise ValueError("The labels must be non-negative.")

    def _n_samples(self, prediction):
        return prediction.numel() // prediction.shape[-1]

    def loss(self, prediction, true_output):
        """
        Calculates the categorical cross entropy with the equations:

        .. math::

                l_i &= -\\sum_{k=1}^K y_{ik}\\cdot\\text{ln}(p_{ik}),\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`p_i = f(x_i)` is the predicted probability distribution, or :math:`p_i = \\text{softmax}(f(x_i))` if from_logits=True, :math:`y_i` is the true distribution and :math:`K` is the number of classes.

        Args:
            prediction (torch.Tensor): A tensor of predicted values as a probability distribution over the last dimension, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values as a probability distribution over the last dimension. Usually one-hot encoded. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check(prediction, true_output)

        log_probability = _log_softmax(prediction) if self.from_logits else torch.log(prediction + 1e-10)
        return self._reduce(-true_output * log_probability, self._n_samples(prediction))

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the categorical cross entropy. With from_logits=True, the gradient with respect to the logits is :math:`\\text{softmax}(z)\\sum_k y_k - y`, which is :math:`\\text{softmax}(z) - y` when the labels sum to one.

        Args:
            prediction (torch.Tensor): A tensor of predicted values as a probability distribution over the last dimension, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values as a probability distribution over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check(prediction, true_output)

        if self.from_logits:
            grad = torch.softmax(prediction, dim=-1) * true_output.sum(dim=-1, keepdim=True) - true_output
        else:
            grad = -true_output / (prediction + 1e-10)
        return self._scale(grad, self._n_samples(prediction))

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the categorical cross entropy loss. With from_logits=True, the diagonal with respect to the logits is :math:`p(1 - p)\\sum_k y_k`, where :math:`p = \\text{softmax}(z)`.

        Args:
            prediction (torch.Tensor): A tensor of predicted values as a probability distribution over the last dimension, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values as a probability distribution over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        self._check(prediction, true_output)

        if self.from_logits:
            probability = torch.softmax(prediction, dim=-1)
            hess = probability * (1 - probability) * true_output.sum(dim=-1, keepdim=True)
        else:
            hess = true_output / (prediction + 1e-10) ** 2
        return self._scale(hess, self._n_samples(prediction))

    def transform_prediction(self, prediction):
        """
        Returns the predicted probabilities. Applies the softmax over the last dimension if from_logits=True, otherwise returns the predictions unchanged.

        Args:
            prediction (torch.Tensor): The raw output of a model.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        return torch.softmax(prediction, dim=-1) if self.from_logits else prediction
