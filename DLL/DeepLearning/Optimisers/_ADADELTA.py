import torch

from ._BaseOptimiser import BaseOptimiser


class ADADELTA(BaseOptimiser):
    """
    The adadelta optimiser. A first order method and therefore does not use information on second gradients, i.e. the hessian matrix. Hence, does not require a lot of memory.

    Args:
        learning_rate (float, optional): The learning rate of the optimiser. Must be positive. Defaults to 0.001.
        rho (float, optional): Determines how long the previous gradients affect the current step direction. Must be in range [0, 1). Defaults to 0.9.
        weight_decay (float, optional): Determines if regularisation should be applied to the weights. Must be in range [0, 1). Defaults to 0.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, weight_decay=0):
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")
        if not isinstance(rho, int | float) or rho < 0 or rho >= 1:
            raise ValueError("rho must be in range [0, 1).")
        if not isinstance(weight_decay, int | float) or weight_decay < 0 or weight_decay >= 1:
            raise ValueError("weight_decay must be in range [0, 1).")

        self.learning_rate = learning_rate
        self.rho = rho
        self.weight_decay = weight_decay
    
    def initialise_parameters(self, model_parameters):
        """
        Initialises the optimiser with the parameters that need to be optimised.

        Args:
            model_parameters (list[torch.Tensor]): The parameters that will be optimised. Must be a list or a tuple of torch tensors.
        """

        if not isinstance(model_parameters, list | tuple):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.square_avg = [torch.zeros_like(param) for param in model_parameters]
        self.accumulate_variables = [torch.zeros_like(param) for param in model_parameters]
        self.t = 0
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """
        self.t += 1
        for i, parameter in enumerate(self.model_parameters):
            if self.weight_decay > 0: parameter.grad += self.weight_decay * parameter
            self.square_avg[i] = self.square_avg[i] * self.rho + parameter.grad ** 2 * (1 - self.rho)
            delta_x = parameter.grad * torch.sqrt(self.accumulate_variables[i] + 1e-10) / torch.sqrt(self.square_avg[i] + 1e-10)
            self.accumulate_variables[i] = self.accumulate_variables[i] * self.rho + delta_x ** 2 * (1 - self.rho)
            parameter -= self.learning_rate * delta_x
