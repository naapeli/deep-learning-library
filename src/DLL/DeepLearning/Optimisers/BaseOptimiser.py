from abc import ABC, abstractmethod

class BaseOptimiser(ABC):
    @abstractmethod
    def initialise_parameters(self, model_parameters):
        pass
    
    @abstractmethod
    def update_parameters(self):
        pass