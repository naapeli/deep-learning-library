import torch

from ._BaseLayer import BaseLayer
from ...Exceptions import NotCompiledError


class LayerList(BaseLayer):
    """
    The list of consecutive layers.

    Args:
        layers (a list of DLL.DeepLearning.Layers.BaseLayer objects): The list of consecutive layers.
    """
    def __init__(self, layers, **kwargs):
        for layer in layers:
            if not isinstance(layer, BaseLayer):
                raise TypeError("layers must be an instances of DLL.DeepLearning.Layers.BaseLayer")
        if kwargs.get("normalisation", None) is not None or kwargs.get("activation", None) is not None:
            raise ValueError("LayerList cannot have normalisation or activation functions directly. Include these functions in the last layer passed to LayerList.")

        super().__init__(None, **kwargs)
        self.name = "Layer list"
        self.layers = layers

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
        for layer in self.layers:
            layer.initialise_layer(input_shape, data_type, device)
            input_shape = layer.output_shape
        self.output_shape = self.layers[-1].output_shape  # all possible layers (even activation layers) now know their output_shapes

    def forward(self, input, training=False, **kwargs):
        """
        Computes the forward values of the input layers.

        Args:
            input (torch.Tensor): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor: The output tensor after the layers with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[-len(self.input_shape):] != self.input_shape:
            raise ValueError(f"Input shape {input.shape[-len(self.input_shape):]} does not match the expected shape {self.input_shape}.")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        self.output = input
        for layer in self.layers:
            self.output = layer.forward(self.output, training=training)
        return self.output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layers. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.

        Returns:
            torch.Tensor of the spesified shape: The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        dCdx = dCdy
        for layer in reversed(self.layers):
            dCdx = layer.backward(dCdx)
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return tuple(param for layer in self.layers for param in layer.get_parameters())
    
    def get_nparams(self):
        n_params = 0
        for layer in self.layers:
            n_params += layer.get_nparams()
        return n_params

    def summary(self, offset=""):
        if not hasattr(self, "input_shape"):
            raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

        summary = super().summary(offset)
        sublayer_offset = offset + "    "
        for layer in self.layers:
            summary += "\n" + offset + layer.summary(sublayer_offset)
        return summary
