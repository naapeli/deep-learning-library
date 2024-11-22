import torch
from math import sqrt

from .Initialiser import Initialiser


class Kaiming_Uniform(Initialiser):
    """
    The Kaiming He uniform initialiser. Kaiming He initialiser should be used for the ReLU or any other activation, which is nonlinear close to origin.

    Args:
        mode (str, optional): Determines if the variance is constant in the forward or backward propagation. If "input_dim", variance is constant in forward propagation, while if "output_dim", the variance is constant in back propagation. Defaults to "input_dim".
    """
    def __init__(self, mode="input_dim"):
        if mode not in ["input_dim", "output_dim"]:
            raise ValueError('mode must be one of "input_dim" or "output_dim".')
        self.mode = mode

    def initialise(self, shape, data_type=torch.float32, device=torch.device("cpu")):
        """
        Initialises a tensor of the wanted shape with values in :math:`U(-a, a)`, where :math:`a = \\sqrt{\\frac{6}{d}}`.

        Args:
            shape (torch.Size): The shape of the wanted tensor.
            data_type (torch.dtype, optional): The data type used in the returned tensor. Defaults to torch.float32.
            device (torch.device, optional): The device of the tensor. Determines if the computation is made using the gpu or the cpu. Defaults to torch.device("cpu").
        """
        input_dim, output_dim = self._get_dims(shape)
        a = sqrt(6/(input_dim if self.mode == "input_dim" else output_dim))
        return 2 * a * torch.rand(size=shape, dtype=data_type, device=device) - a


class Kaiming_Normal(Initialiser):
    """
    The Kaiming He normal initialiser. Kaiming He initialiser should be used for the ReLU or any other activation, which is nonlinear close to origin.

    Args:
        mode (str, optional): Determines if the variance is constant in the forward or backward propagation. If "input_dim", variance is constant in forward propagation, while if "output_dim", the variance is constant in back propagation. Defaults to "input_dim".
    """
    def __init__(self, mode="input_dim"):
        if mode not in ["input_dim", "output_dim"]:
            raise ValueError('mode must be one of "input_dim" or "output_dim".')
        self.mode = mode

    def initialise(self, shape, data_type=torch.float32, device=torch.device("cpu")):
        """
        Initialises a tensor of the wanted shape with values in :math:`N(0, \\sigma^2)`, where :math:`\\sigma = \\sqrt{\\frac{2}{d_{\\text{input}} + d_{\\text{output}}}}`.

        Args:
            shape (torch.Size): The shape of the wanted tensor.
            data_type (torch.dtype, optional): The data type used in the returned tensor. Defaults to torch.float32.
            device (torch.device, optional): The device of the tensor. Determines if the computation is made using the gpu or the cpu. Defaults to torch.device("cpu").
        """
        input_dim, output_dim = self._get_dims(shape)
        return torch.normal(mean=0, std=sqrt(2/(input_dim if self.mode == "input_dim" else output_dim)), size=shape, dtype=data_type, device=device)
