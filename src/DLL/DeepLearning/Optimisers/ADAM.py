import torch

from .BaseOptimiser import BaseOptimiser


class Adam(BaseOptimiser):
    """
    The adaptive moment estimation optimiser. Is very robust and does not require a lot of tuning it's hyperparameters. A first order method and therefore does not use information on second gradients, i.e. the hessian matrix. Hence, does not require a lot of memory. Is based on algorithm 1 on `this paper <https://arxiv.org/pdf/1412.6980>`_.

    Args:
        learning_rate (float, optional): The learning rate of the optimiser. Must be positive. Defaults to 0.001.
        beta1 (float, optional): Determines how long the previous gradients affect the current step direction. Must be in range [0, 1). Defaults to 0.9.
        beta2 (float, optional): Determines how long the previous squared gradients affect the current step direction. Must be in range [0, 1). Defaults to 0.999.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")
        if not isinstance(beta1, int | float) or beta1 < 0 or beta1 >= 1:
            raise ValueError("momentum must be in range [0, 1).")
        if not isinstance(beta2, int | float) or beta2 < 0 or beta2 >= 1:
            raise ValueError("momentum must be in range [0, 1).")
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
    
    def initialise_parameters(self, model_parameters):
        """
        Initialises the optimiser with the parameters that need to be optimised.

        Args:
            model_parameters (list[torch.Tensor]): The parameters that will be optimised. Must be a list or a tuple of torch tensors.
        """

        if not isinstance(model_parameters, list | tuple):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.m = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.model_parameters]
        self.t = 0
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """

        self.t += 1
        for i, parameter in enumerate(self.model_parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * parameter.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * parameter.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            parameter -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-10)
