import torch

from abc import ABC, abstractmethod


class BaseOptimiser(ABC):
    """
    :meta private:
    """
    @abstractmethod
    def initialise_parameters(self, model_parameters):
        pass
    
    @abstractmethod
    def update_parameters(self):
        pass

    def zero_grad(self):
        for param in self.model_parameters:
            param.grad = torch.zeros_like(param)
