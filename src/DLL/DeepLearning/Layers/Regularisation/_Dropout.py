import torch

from ._BaseRegularisation import BaseRegularisation


class Dropout(BaseRegularisation):
    """
    The dropout layer for neural networks.

    Args:
        p (float, optional): The probability of a node being dropped out. Must be strictly between 0 and 1. Defaults to 0.5.
    """
    def __init__(self, p=0.5, **kwargs):
        if p <= 0 or 1 <= p:
            raise ValueError("p must be strictly between 0 and 1.")
        
        super().__init__(**kwargs)
        self.p = 1 - p
        self.name = "Dropout"
    
    def initialise_layer(self, **kwargs):
        """
        :meta private:
        """

        super().initialise_layer(**kwargs)

        self.mask = torch.rand(size=self.output_shape, dtype=self.data_type, device=self.device) < self.p

    def forward(self, input, training=False, **kwargs):
        """
        Sets some values of the input to zero with probability p.

        Args:
            input (torch.Tensor of shape (batch_size, channels, ...)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor: The output tensor after the transformation with the same shape as the input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")
        
        self.input = input
        if training:
            self.mask = torch.rand(size=self.input.shape, dtype=self.data_type, device=self.device) < self.p
            self.output = self.input * self.mask / self.p
        else:
            self.output = self.input
        return self.output
    
    def backward(self, dCdy, training=False, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (batch_size, channels, ...): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        dCdx = dCdy * self.mask if training else dCdy
        return dCdx
    
    def summary(self):
        """
        :meta private:
        """
        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        return f"{self.name} - Output: ({output_shape}) - Keep probability: {self.p}"
