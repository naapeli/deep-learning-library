import torch

from .BaseLoss import BaseLoss


"""
Calculates the categorical cross entropy and the gradient between a prediction vector as well as intended outputs
"""
class cce(BaseLoss):
    def __init__(self, reduction="mean"):
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        if self.reduction == "mean":
            return -torch.mean(true_output.reshape(prediction.shape) * torch.log(prediction + 1e-5))
        return -torch.sum(true_output.reshape(prediction.shape) * torch.log(prediction + 1e-5))

    def gradient(self, prediction, true_output):
        if self.reduction == "mean":
            return -true_output.reshape(prediction.shape) / ((prediction + 1e-5) * prediction.shape[0])
        return -true_output.reshape(prediction.shape) / (prediction + 1e-5)
