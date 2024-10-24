import torch


"""
Calculates the exponential loss and the gradient between a prediction vector as well as intended outputs. Meant for binary classification.
"""
class exponential:
    def loss(self, prediction, true_output):
        # assert set(torch.unique(true_output).numpy()) == {-1, 1}, "The classes must be labelled -1 and 1."
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        # return torch.exp(-true_output * prediction)
        true_output = true_output.float()
        loss = true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
        return torch.mean(loss)

    def gradient(self, prediction, true_output):
        # assert set(torch.unique(true_output).numpy()) == {-1, 1}, "The classes must be labelled -1 and 1."
        assert set(torch.unique(true_output).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        # return -true_output * self.loss(prediction, true_output)
        true_output = true_output.float()
        return -true_output * torch.exp(-prediction) + (1 - true_output) * torch.exp(prediction)
