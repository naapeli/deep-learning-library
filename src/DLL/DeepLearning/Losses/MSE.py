import torch


"""
Calculates the mean squared error and the gradient between a prediction vector as well as intended outputs
"""
class mse:
    def __init__(self, reduction="mean"):
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        if self.reduction == "mean":
            return torch.mean((prediction - true_output.reshape(prediction.shape)) ** 2)
        return torch.sum((prediction - true_output.reshape(prediction.shape)) ** 2)

    def gradient(self, prediction, true_output):
        if self.reduction == "mean":
            return 2 * (prediction - true_output.reshape(prediction.shape)) / prediction.shape[0]
        return 2 * (prediction - true_output.reshape(prediction.shape))
