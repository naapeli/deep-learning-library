import torch

from .BaseLoss import BaseLoss


"""
Calculates the binary cross entropy and the gradient between a prediction vector as well as intended outputs
"""
class bce(BaseLoss):
    def __init__(self, reduction="mean"):
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        if self.reduction == "mean":
            return -torch.mean(true_output.reshape(prediction.shape) * torch.log(prediction + 1e-10) + (1 - true_output.reshape(prediction.shape)) * torch.log(1 - prediction + 1e-10))
        return -torch.sum(true_output.reshape(prediction.shape) * torch.log(prediction + 1e-10) + (1 - true_output.reshape(prediction.shape)) * torch.log(1 - prediction + 1e-10))

    def gradient(self, prediction, true_output):
        if self.reduction == "mean":
            return (prediction - true_output) / ((prediction * (1 - prediction) + 1e-10) * prediction.shape[0])
        return (prediction - true_output) / (prediction * (1 - prediction) + 1e-10)
