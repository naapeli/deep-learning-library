import torch

from .BaseOptimiser import BaseOptimiser


class sgd(BaseOptimiser):
    """
    Stochastic gradient descent optimiser with momentum. A first order method and therefore does not use information on second gradients, i.e. the hessian matrix. Hence, does not require a lot of memory.

    Args:
        learning_rate (float, optional): The learning rate of the optimiser. Must be positive. Defaults to 0.001.
        momentum (float, optional): Determines how long the previous gradients affect the current direction. Must be in range [0, 1). Defaults to 0.9.
    """
    def __init__(self, learning_rate=0.001, momentum=0.9):
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")
        if not isinstance(momentum, int | float) or momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in range [0, 1).")

        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def initialise_parameters(self, model_parameters):
        """
        Initialises the optimiser with the parameters that need to be optimised.

        Args:
            model_parameters (list[torch.Tensor]): The parameters that will be optimised. Must be a list or a tuple of torch tensors.
        """

        if not isinstance(model_parameters, list | tuple):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.changes = [torch.zeros_like(parameter) for parameter in self.model_parameters]
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """

        for i, parameter in enumerate(self.model_parameters):
            change = self.learning_rate * parameter.grad + self.momentum * self.changes[i]
            parameter -= change
            self.changes[i] = change
