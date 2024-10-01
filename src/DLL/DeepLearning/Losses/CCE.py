import torch


"""
Calculates the categorical cross entropy and the gradient between a prediction vector as well as intended outputs
"""
class cce:
    def loss(self, prediction, true_output):
        return -torch.mean(true_output.reshape(prediction.shape) * torch.log(prediction + 1e-5))

    def gradient(self, prediction, true_output):
        return -true_output.reshape(prediction.shape) / (prediction + 1e-5)
