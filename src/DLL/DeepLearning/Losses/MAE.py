import torch


class mae:
    def loss(self, prediction, true_output):
        return torch.abs(prediction - true_output.reshape(prediction.shape)).mean()

    def gradient(self, prediction, true_output):
        return torch.sign(prediction - true_output.reshape(prediction.shape))
