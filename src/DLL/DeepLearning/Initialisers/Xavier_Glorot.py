import torch
from math import sqrt

from .Initialiser import Initialiser


class Xavier_Uniform(Initialiser):
    """
    The Xavier Glorot uniform initialiser. Xavier Glorot initialiser should be used for tanh, sigmoid, softmax or other activations, which are approximately linear close to origin.
    """
    def initialise(self, shape, data_type=torch.float32, device=torch.device("cpu")):
        """
        Initialises a tensor of the wanted shape with values in :math:`U(-a, a)`, where :math:`a = \\sqrt{\\frac{6}{d_{\\text{input}} + d_{\\text{output}}}}`.

        Args:
            shape (torch.Size): The shape of the wanted tensor.
            data_type (torch.dtype, optional): The data type used in the returned tensor. Defaults to torch.float32.
            device (torch.device, optional): The device of the tensor. Determines if the computation is made using the gpu or the cpu. Defaults to torch.device("cpu").
        """
        input_dim, output_dim = self._get_dims(shape)
        a = sqrt(6/(input_dim + output_dim))
        return 2 * a * torch.rand(size=shape, dtype=data_type, device=device) - a


class Xavier_Normal(Initialiser):
    """
    The Xavier Glorot normal initialiser. Xavier Glorot initialiser should be used for tanh, sigmoid, softmax or other activations, which are approximately linear close to origin.
    """
    def initialise(self, shape, data_type=torch.float32, device=torch.device("cpu")):
        """
        Initialises a tensor of the wanted shape with values in :math:`N(0, \\sigma^2)`, where :math:`\\sigma = \\sqrt{\\frac{2}{d_{\\text{input}} + d_{\\text{output}}}}`.

        Args:
            shape (torch.Size): The shape of the wanted tensor.
            data_type (torch.dtype, optional): The data type used in the returned tensor. Defaults to torch.float32.
            device (torch.device, optional): The device of the tensor. Determines if the computation is made using the gpu or the cpu. Defaults to torch.device("cpu").
        """
        input_dim, output_dim = self._get_dims(shape)
        return torch.normal(mean=0, std=sqrt(2/(input_dim + output_dim)), size=shape, dtype=data_type, device=device)
