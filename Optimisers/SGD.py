import torch


"""
Implements stochastic gradient descent optimiser with momentum
"""
class sgd:
    def __init__(self, model_parameters, learning_rate=0.001, momentum=0.9):
        assert 0 <= momentum and momentum < 1, "momentum should be in range [0, 1)"
        self.model_parameters = model_parameters
        self.changes = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update_parameters(self):
        for i, parameter in enumerate(self.model_parameters):
            change = self.learning_rate * parameter.grad + self.momentum * self.changes[i]
            parameter -= change
            self.changes[i] = change
