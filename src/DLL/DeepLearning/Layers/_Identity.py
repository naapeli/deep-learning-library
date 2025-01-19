import torch

from .Activations._Activation import Activation


class Identity(Activation):
    """
    The identity layer.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.name = "Identity"

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if not isinstance(input_shape, tuple | list):
            raise ValueError("input_shape must be a tuple.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        super().initialise_layer(input_shape, data_type, device)

    def forward(self, input, training=False, **kwargs):
        """
        Returns the input.

        Args:
            input (torch.Tensor of shape (n_samples, n_features)): The input to the dense layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor: The same tensor as the input
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[-len(self.input_shape):] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[-len(self.input_shape):], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        return input

    def backward(self, dCdy, **kwargs):
        """
        Returns the gradient.

        Args:
            dCdy (torch.Tensor): The gradient given by the next layer.
            
        Returns:
            torch.Tensor: The same tensor as the input gradient
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape and not (dCdy.ndim == 1 and self.output_shape[0] == 0):
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")
        
        return dCdy
