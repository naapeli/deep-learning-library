from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """
    :meta private:
    """
    @abstractmethod
    def loss(self, prediction, true_output):
        pass
    
    @abstractmethod
    def gradient(self, prediction, true_output):
        pass

    @abstractmethod
    def hessian(self, prediction, true_output):
        pass
