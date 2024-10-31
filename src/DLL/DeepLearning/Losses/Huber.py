import torch

from .BaseLoss import BaseLoss


class Huber(BaseLoss):
    def __init__(self, delta=1.0, reduction="mean"):
        self.delta = delta
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        error = prediction - true_output.reshape(prediction.shape)
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        if self.reduction == "mean":
            return torch.where(abs_error <= self.delta, quadratic, linear).mean()
        return torch.where(abs_error <= self.delta, quadratic, linear).sum()

    def gradient(self, prediction, true_output):
        error = prediction - true_output.reshape(prediction.shape)
        abs_error = torch.abs(error)
        quadratic_grad = error
        linear_grad = self.delta * torch.sign(error)
        if self.reduction == "mean":
            return torch.where(abs_error <= self.delta, quadratic_grad, linear_grad) / prediction.shape[0]
        return torch.where(abs_error <= self.delta, quadratic_grad, linear_grad)
