from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    """
    :meta private:
    """
    @abstractmethod
    def loss(self, prediction, true_output):
        pass

    @abstractmethod
    def gradient(self, prediction, true_output):
        pass

    @abstractmethod
    def hessian(self, prediction, true_output):
        pass

    def transform_prediction(self, prediction):
        """
        Maps the raw output of a model to the form the metrics expect. The identity for every loss except the ones constructed with ``from_logits=True``, which return the probabilities given by the sigmoid or the softmax of the logits.

        Args:
            prediction (torch.Tensor): The raw output of a model.

        Returns:
            torch.Tensor: The transformed predictions.
        """
        return prediction

    def _check_inputs(self, prediction, true_output):
        if not isinstance(prediction, torch.Tensor) or not isinstance(true_output, torch.Tensor):
            raise TypeError("prediction and true_output must be torch tensors.")
        if prediction.shape != true_output.shape:
            raise ValueError("prediction and true_output must have the same shape.")

    def _reduce(self, values, n):
        if self.reduction == "mean":
            return values.sum() / n
        return values.sum()

    def _scale(self, values, n):
        if self.reduction == "mean":
            return values / n
        return values


def _check_reduction(reduction):
    if reduction not in ["mean", "sum"]:
        raise ValueError('reduction must be in ["mean", "sum"].')


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _check_soft_labels(true_output):
    if torch.any(true_output < 0) or torch.any(true_output > 1):
        raise ValueError("The labels must be in the range [0, 1].")


def _check_binary_labels(true_output):
    if torch.any((true_output != 0) & (true_output != 1)):
        raise ValueError("The classes must be labelled 0 and 1.")


def _log_sigmoid(logits):
    # log(sigmoid(z)) = -softplus(-z) = min(z, 0) - log(1 + exp(-|z|)), which never overflows
    return torch.clamp(logits, max=0) - torch.log1p(torch.exp(-torch.abs(logits)))


def _log_softmax(logits):
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)
