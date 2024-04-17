import torch


class mse:
    def loss(self, prediction, true_output):
        return torch.mean((prediction - true_output) ** 2)

    def gradient(self, prediction, true_output):
        return 2 * (prediction - true_output) / prediction.shape[0]
