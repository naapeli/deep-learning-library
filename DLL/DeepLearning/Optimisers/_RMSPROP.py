import torch

from ._BaseOptimiser import BaseOptimiser


class RMSPROP(BaseOptimiser):
    """
    The adaptive moment estimation optimiser. Is very robust and does not require a lot of tuning it's hyperparameters. A first order method and therefore does not use information on second gradients, i.e. the hessian matrix. Hence, does not require a lot of memory.

    Args:
        learning_rate (float, optional): The learning rate of the optimiser. Must be positive. Defaults to 0.001.
        alpha (float, optional): A smoothing constant. Defaults to 0.99.
        momentum (float, optional): Determines how long the previous gradients affect the current direction. Must be in range [0, 1). Defaults to 0.
        weight_decay (float, optional): Determines if regularisation should be applied to the weights. Must be in range [0, 1). Defaults to 0.
        centered (bool, optional): Determines if a centered version of the algorithm is used. Defaults to False.
    """
    def __init__(self, learning_rate=0.001, alpha=0.99, momentum=0, weight_decay=0, centered=False):
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")
        if not isinstance(momentum, int | float) or momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in range [0, 1).")
        if not isinstance(weight_decay, int | float) or weight_decay < 0 or weight_decay >= 1:
            raise ValueError("weight_decay must be in range [0, 1).")
        
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered
    
    def initialise_parameters(self, model_parameters):
        """
        Initialises the optimiser with the parameters that need to be optimised.

        Args:
            model_parameters (list[torch.Tensor]): The parameters that will be optimised. Must be a list or a tuple of torch tensors.
        """

        if not isinstance(model_parameters, list | tuple):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.square_average = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.buffer = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.g_average = [torch.zeros_like(parameter) for parameter in self.model_parameters]
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """
        for i, parameter in enumerate(self.model_parameters):
            if self.weight_decay > 0: parameter.grad += self.weight_decay * parameter
            self.square_average[i] = self.alpha * self.square_average[i] + (1 - self.alpha) * parameter.grad ** 2
            v_tilde = self.square_average[i]
            if self.centered:
                self.g_average[i] = self.alpha * self.g_average[i] + (1 - self.alpha) * parameter.grad
                v_tilde -= self.g_average[i] ** 2
            if self.momentum > 0:
                self.buffer[i] = self.momentum * self.buffer[i] + parameter.grad / (torch.sqrt(v_tilde) + 1e-10)
                parameter -= self.learning_rate * self.buffer[i]
            else:
                parameter -= self.learning_rate * parameter.grad / (torch.sqrt(v_tilde) + 1e-10)
