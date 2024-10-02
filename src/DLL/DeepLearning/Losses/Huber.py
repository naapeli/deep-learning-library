import torch


class Huber:
    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, prediction, true_output):
        error = prediction - true_output.reshape(prediction.shape)
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        return torch.where(abs_error <= self.delta, quadratic, linear).mean()

    def gradient(self, prediction, true_output):
        error = prediction - true_output.reshape(prediction.shape)
        abs_error = torch.abs(error)
        quadratic_grad = error
        linear_grad = self.delta * torch.sign(error)
        return torch.where(abs_error <= self.delta, quadratic_grad, linear_grad)
