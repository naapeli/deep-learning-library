import torch

from .BaseOptimiser import BaseOptimiser


"""
Implements the ADAM optimiser
"""
class Adam(BaseOptimiser):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        assert 0 <= beta1 and beta1 < 1, "beta1 should be in range [0, 1)"
        assert 0 <= beta2 and beta2 < 1, "beta2 should be in range [0, 1)"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
    
    def initialise_parameters(self, model_parameters):
        self.model_parameters = model_parameters
        self.m = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.t = 0
    
    def update_parameters(self):
        self.t += 1
        for i, parameter in enumerate(self.model_parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * parameter.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * parameter.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            parameter -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-10)
