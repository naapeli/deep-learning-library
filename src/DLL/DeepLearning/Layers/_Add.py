import torch

from ._BaseLayer import BaseLayer
from ...Exceptions import NotCompiledError
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation


class Add(BaseLayer):
    """
    The addition layer.

    Args:
        layer1 (DLL.DeepLearning.Layers.BaseLayer object): The first layer the input is passed to. The results of each layer are added together. The input and outpput shapes of the layers must be the same.
        layer2 (DLL.DeepLearning.Layers.BaseLayer object): The second layer the input is passed to. The results of each layer are added together. The input and outpput shapes of the layers must be the same.
    """
    def __init__(self, layer1, layer2, activation=None, normalisation=None, **kwargs):
        if not isinstance(layer1, BaseLayer) or not isinstance(layer2, BaseLayer):
            raise TypeError("layers must be an instances of DLL.DeepLearning.Layers.BaseLayer")
        if layer1.output_shape != layer2.output_shape and layer1.output_shape is not None and layer2.output_shape is not None:
            raise ValueError("Layers must have the same output shape.")
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__(layer1.output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "Add"
        self.layer1 = layer1
        self.layer2 = layer2

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
        
        if self.layer1.output_shape is None and input_shape != self.layer2.output_shape:
            raise ValueError(f"Layers must have the same output shape {input_shape} vs {self.layer2.output_shape}.")
        if self.layer2.output_shape is None and input_shape != self.layer1.output_shape:
            raise ValueError(f"Layers must have the same output shape {input_shape} vs {self.layer1.output_shape}.")

        self.layer1.initialise_layer(input_shape, data_type, device)
        self.layer2.initialise_layer(input_shape, data_type, device)
        super().initialise_layer(input_shape, data_type, device)

    def forward(self, input, training=False, **kwargs):
        """
        Computes the forward values of the input layers and adds them together.

        Args:
            input (torch.Tensor): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor: The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[-len(self.input_shape):] != self.input_shape:
            raise ValueError(f"Input shape {input.shape[-len(self.input_shape):]} does not match the expected shape {self.input_shape}.")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        val1 = self.layer1.forward(input, training=training)
        val2 = self.layer2.forward(input.flip(1), training=training)
        self.output = val1 + val2
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
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
        
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        dCdx1 = self.layer1.backward(dCdy, **kwargs)
        dCdx2 = self.layer2.backward(dCdy, **kwargs)

        dCdx = dCdx1 + dCdx2
        return dCdx

    def get_parameters(self):
        """
        :meta private:
        """
        return (*self.layer1.get_parameters(), *self.layer2.get_parameters())
    
    def get_nparams(self):
        return self.layer1.get_nparams() + self.layer2.get_nparams()
    
    def summary(self, offset=""):
        if not hasattr(self, "input_shape"):
            raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

        sublayer_offset = offset + "    "
        summary1 = "\n" + self.layer1.summary(sublayer_offset)
        summary2 = "\n" + self.layer2.summary(sublayer_offset)
        return super().summary(offset) + summary1 + summary2
