import torch


"""
Calculates the binary cross entropy and the gradient between a prediction vector as well as intended outputs
"""
class bce:
    def loss(self, prediction, true_output):
        return -torch.mean(true_output.reshape(prediction.shape) * torch.log2(prediction + 1e-10) - (1 - true_output.reshape(prediction.shape)) * torch.log2(1 - prediction + 1e-10))

    def gradient(self, prediction, true_output):
        return (prediction - true_output) / (prediction * (1 - prediction) + 1e-10)
