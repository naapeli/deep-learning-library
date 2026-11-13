import torch

from ._BaseLoss import BaseLoss, _check_reduction, _check_bool, _check_soft_labels, _log_sigmoid, _log_softmax


class Focal(BaseLoss):
    """
    The focal loss from `this paper <https://arxiv.org/abs/1708.02002>`_. A cross entropy loss, where the loss of every sample is multiplied by :math:`(1 - p_t)^\\gamma`, where :math:`p_t` is the predicted probability of the true class. This down-weights the easy, well classified samples, so that the training focuses on the hard ones. Useful for imbalanced classification. With gamma=0 and alpha=None, it is identical to the binary or the categorical cross entropy.

    Args:
        gamma (int | float, optional): The focusing parameter :math:`\\gamma`. Must be a non-negative real number. Defaults to 2.
        alpha (float | torch.Tensor | None, optional): The class weights. If categorical=False, must be a float in range (0, 1), which is the weight of the positive class, while the negative class gets the weight 1 - alpha. If categorical=True, must be a positive float or a tensor of shape (n_classes,) of positive weights. If None, every class has the weight 1. Defaults to None.
        from_logits (bool, optional): If True, the predictions are logits, which are mapped to probabilities with the sigmoid (or softmax if categorical=True) inside the loss. Defaults to False.
        categorical (bool, optional): If False, every element is an independent binary problem as in :class:`BCE`. If True, the last dimension is the class dimension as in :class:`CCE`. Defaults to False.
        reduction (str, optional): The reduction method. Must be one of "mean" or "sum". Defaults to "mean".

    Note:
        The labels can be soft as in :class:`BCE` and :class:`CCE`. The gradient is calculated analytically, while the diagonal of the hessian is calculated with automatic differentiation.
    """
    def __init__(self, gamma=2, alpha=None, from_logits=False, categorical=False, reduction="mean"):
        if not isinstance(gamma, int | float) or gamma < 0:
            raise ValueError("gamma must be a non-negative real number.")
        _check_bool(from_logits, "from_logits")
        _check_bool(categorical, "categorical")
        _check_reduction(reduction)
        if not categorical and alpha is not None and (not isinstance(alpha, float) or alpha <= 0 or alpha >= 1):
            raise ValueError("With categorical=False, alpha must be None or a float in range (0, 1).")
        if categorical and alpha is not None:
            if isinstance(alpha, int | float):
                if alpha <= 0:
                    raise ValueError("alpha must be positive.")
            elif not isinstance(alpha, torch.Tensor) or alpha.ndim != 1 or torch.any(alpha <= 0):
                raise ValueError("With categorical=True, alpha must be None, a positive real number or a 1 dimensional tensor of positive weights.")

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.categorical = categorical
        self.reduction = reduction
        self._eps = 1e-7

    def _check(self, prediction, true_output):
        self._check_inputs(prediction, true_output)
        if self.categorical:
            if torch.any(true_output < 0):
                raise ValueError("The labels must be non-negative.")
            if isinstance(self.alpha, torch.Tensor) and len(self.alpha) != prediction.shape[-1]:
                raise ValueError("alpha must have one weight for every class.")
        else:
            _check_soft_labels(true_output)

    def _n_samples(self, prediction):
        return prediction.numel() // prediction.shape[-1] if self.categorical else prediction.numel()

    def _weights(self, prediction):
        # the weights of the positive and the negative term
        if self.categorical:
            if isinstance(self.alpha, torch.Tensor):
                return self.alpha.to(prediction.dtype), None
            return (1.0 if self.alpha is None else self.alpha), None
        if self.alpha is None:
            return 1.0, 1.0
        return self.alpha, 1 - self.alpha

    def _log_probabilities(self, prediction):
        # returns (p, log p, log(1 - p)) for binary and (p, log p) for categorical predictions
        if self.from_logits:
            if self.categorical:
                log_p = _log_softmax(prediction)
                return log_p.exp(), log_p, None
            log_p = _log_sigmoid(prediction)
            return log_p.exp(), log_p, _log_sigmoid(-prediction)
        p = prediction.clamp(self._eps, 1 - self._eps)
        return p, torch.log(p), torch.log(1 - p)

    def _elementwise_loss(self, prediction, true_output):
        positive_weight, negative_weight = self._weights(prediction)
        p, log_p, log_1mp = self._log_probabilities(prediction)
        loss = -positive_weight * true_output * (1 - p) ** self.gamma * log_p
        if not self.categorical:
            loss = loss - negative_weight * (1 - true_output) * p ** self.gamma * log_1mp
        return loss

    def loss(self, prediction, true_output):
        """
        Calculates the focal loss with the equations:

        .. math::

                l_i &= -\\alpha y_i (1 - p_i)^\\gamma\\text{ln}(p_i) - (1 - \\alpha)(1 - y_i) p_i^\\gamma\\text{ln}(1 - p_i),\\\\
                L_{sum} &= \\sum_{i=1}^n l_i \\text{ or } L_{mean} = \\frac{1}{n}\\sum_{i=1}^n l_i,

        where :math:`p_i` is the predicted probability and :math:`y_i` is the true value. With categorical=True,

        .. math::

                l_i = -\\sum_{k=1}^K \\alpha_k y_{ik}(1 - p_{ik})^\\gamma\\text{ln}(p_{ik}).

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. If categorical=True, a probability distribution over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor containing a single value with the loss.
        """
        self._check(prediction, true_output)

        return self._reduce(self._elementwise_loss(prediction, true_output), self._n_samples(prediction))

    def gradient(self, prediction, true_output):
        """
        Calculates the gradient of the focal loss with respect to the predictions (the probabilities, or the logits if from_logits=True).

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. If categorical=True, a probability distribution over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the gradients.
        """
        self._check(prediction, true_output)

        positive_weight, negative_weight = self._weights(prediction)
        p, log_p, log_1mp = self._log_probabilities(prediction)
        gamma = self.gamma
        one_minus_p = (1 - p).clamp(min=1e-12)  # avoids 0 ** (gamma - 1) for gamma < 1

        if self.categorical:
            # p_k * dl/dp_k
            h = -positive_weight * true_output * (one_minus_p ** gamma - gamma * p * one_minus_p ** (gamma - 1) * log_p)
            if self.from_logits:
                # chain rule through the softmax: dl/dz_j = p_j * (dl/dp_j - sum_k p_k * dl/dp_k)
                grad = h - p * h.sum(dim=-1, keepdim=True)
            else:
                grad = h / p
        elif self.from_logits:
            # dl/dz = dl/dp * p * (1 - p)
            grad = positive_weight * true_output * one_minus_p ** gamma * (gamma * p * log_p - (1 - p)) \
                + negative_weight * (1 - true_output) * p ** gamma * (p - gamma * (1 - p) * log_1mp)
        else:
            grad = -positive_weight * true_output * (one_minus_p ** gamma / p - gamma * one_minus_p ** (gamma - 1) * log_p) \
                - negative_weight * (1 - true_output) * (gamma * p ** (gamma - 1) * log_1mp - p ** gamma / one_minus_p)
        if not self.from_logits:
            outside = (prediction < self._eps) | (prediction > 1 - self._eps)
            grad = torch.where(outside, 0.0, grad)  # the predictions are clipped to [eps, 1 - eps] in the loss
        return self._scale(grad, self._n_samples(prediction))

    def hessian(self, prediction, true_output):
        """
        Calculates the diagonal of the hessian matrix of the focal loss with automatic differentiation.

        Args:
            prediction (torch.Tensor): A tensor of predicted probabilities, or logits if from_logits=True. Must be the same shape as the true_output.
            true_output (torch.Tensor): A tensor of true values in range [0, 1]. If categorical=True, a probability distribution over the last dimension. Must be the same shape as the prediction.

        Returns:
            torch.Tensor: A tensor of the same shape as the inputs containing the diagonal of the hessian matrix.
        """
        self._check(prediction, true_output)

        with torch.enable_grad():
            x = prediction.detach().clone().requires_grad_(True)
            grad = torch.autograd.grad(self._elementwise_loss(x, true_output).sum(), x, create_graph=True)[0]
            if self.categorical and self.from_logits:
                # the softmax couples the classes, so every class needs its own backward pass
                hess = torch.zeros_like(prediction)
                for k in range(prediction.shape[-1]):
                    hess[..., k] = torch.autograd.grad(grad[..., k].sum(), x, retain_graph=True)[0][..., k]
            else:
                # the loss is a sum of terms that each depend on a single element
                hess = torch.autograd.grad(grad.sum(), x)[0]
        return self._scale(hess.detach(), self._n_samples(prediction))

    def transform_prediction(self, prediction):
        """
        Returns the predicted probabilities. Applies the sigmoid (or the softmax if categorical=True) if from_logits=True, otherwise returns the predictions unchanged.

        Args:
            prediction (torch.Tensor): The raw output of a model.

        Returns:
            torch.Tensor: The predicted probabilities.
        """
        if not self.from_logits:
            return prediction
        return torch.softmax(prediction, dim=-1) if self.categorical else torch.sigmoid(prediction)
