import torch


"""
Calculates the exponential loss and the gradient between a prediction vector as well as intended outputs. Meant for binary classification.
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_loss/loss.py#L268 - ExponentialLoss
"""
class exponential:
    def __init__(self, reduction="mean"):
        # reduction in ["mean", "sum"]
        self.reduction = reduction

    def loss(self, prediction, true_output):
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        true_output = true_output.float()
        loss = true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def gradient(self, prediction, true_output):
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        true_output = true_output.float()
        if self.reduction == "mean":
            return (-true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)) / prediction.shape[0]
        return -true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
