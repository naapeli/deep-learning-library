import torch

from .BaseLoss import BaseLoss


class mae(BaseLoss):
    def __init__(self, reduction="mean"):
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        if self.reduction == "mean":
            return torch.abs(prediction - true_output.reshape(prediction.shape)).mean()
        return torch.abs(prediction - true_output.reshape(prediction.shape)).sum()
        

    def gradient(self, prediction, true_output):
        if self.reduction == "mean":
            return torch.sign(prediction - true_output.reshape(prediction.shape)) / prediction.shape[0]
        return torch.sign(prediction - true_output.reshape(prediction.shape))
