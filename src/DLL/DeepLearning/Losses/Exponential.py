import torch


"""
Calculates the exponential loss and the gradient between a prediction vector as well as intended outputs. Meant for binary classification.
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_loss/loss.py#L268 - ExponentialLoss
"""
class exponential:
    def loss(self, prediction, true_output):
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        true_output = true_output.float()
        loss = true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
        return torch.mean(loss)

    def gradient(self, prediction, true_output):
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        true_output = true_output.float()
        return -true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
