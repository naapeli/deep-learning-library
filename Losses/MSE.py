import torch


class mse:
    def loss(self, prediction, true_output):
        return torch.mean((prediction - true_output.reshape(prediction.shape)) ** 2)

    def gradient(self, prediction, true_output):
        return 2 * (prediction - true_output.reshape(prediction.shape)) / prediction.shape[0]
