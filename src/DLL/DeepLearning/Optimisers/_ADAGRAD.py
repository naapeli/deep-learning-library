import torch

from ._BaseOptimiser import BaseOptimiser


class ADAGRAD(BaseOptimiser):
    """
    The adaptive gradient optimiser. A first order method and therefore does not use information on second gradients, i.e. the hessian matrix. Hence, does not require a lot of memory.

    Args:
        learning_rate (float, optional): The learning rate of the optimiser. Must be positive. Defaults to 0.001.
        lr_decay (float, optional): Determines how fast the learning rate decreases. Must be positive. Defaults to 0.
        weight_decay (float, optional): Determines if regularisation should be applied to the weights. Must be in range [0, 1). Defaults to 0.
    """
    def __init__(self, learning_rate=0.001, lr_decay=0, weight_decay=0):
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")
        if not isinstance(lr_decay, int | float) or lr_decay < 0:
            raise ValueError("lr_decay must be positive.")
        if not isinstance(weight_decay, int | float) or weight_decay < 0 or weight_decay >= 1:
            raise ValueError("weight_decay must be in range [0, 1).")
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
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
        self.state_sum = [torch.zeros_like(param) for param in model_parameters]
        self.t = 0
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """
        self.t += 1
        for i, parameter in enumerate(self.model_parameters):
            learning_rate = self.learning_rate / (1 + (self.t - 1) * self.lr_decay)
            if self.weight_decay > 0: parameter.grad += self.weight_decay * parameter
            self.state_sum[i] += parameter.grad ** 2
            parameter -= learning_rate * parameter.grad / (torch.sqrt(self.state_sum[i]) + 1e-10)
