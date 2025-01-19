import torch
from copy import deepcopy

from ._BaseLayer import BaseLayer
from . import RNN, LSTM
from ...Exceptions import NotCompiledError


class Bidirectional(BaseLayer):
    """
    The bidirectional wrapper for LSTM or RNN layers.

    Args:
        layer (DLL.DeepLearning.Layers.RNN or LSTM object): The input is passed to this layer in forward and reverse. The results of each layer are concatanated together along the feature axis.
    """
    def __init__(self, layer, **kwargs):
        if not isinstance(layer, RNN) and not isinstance(layer, LSTM):
            raise TypeError("layer must be an instance of DLL.DeepLearning.Layers.RNN or LSTM")

        # Change the layers output shape to have at least 1 dimensional features.
        if layer.output_shape[0] == 0:
            layer.output_shape = (1,)
        output_shape = (2 * layer.output_shape[0],)

        super().__init__(output_shape, **kwargs)
        self.name = "Bidirectional"
        self.forward_layer = layer
        self.backward_layer = deepcopy(layer)

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if len(input_shape) == 2: input_shape = (input_shape[1],)  # Include only the number of features
        if not isinstance(input_shape, tuple | list) or len(input_shape) != 1:
            raise ValueError("input_shape must be a tuple of length 1.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')
    
        self.forward_layer.initialise_layer(input_shape, data_type, device)
        self.backward_layer.initialise_layer(input_shape, data_type, device)

        super().initialise_layer(input_shape, data_type, device)

    def forward(self, input, training=False, **kwargs):
        """
        Computes the forward values of the RNN or LSTM layer for both normal input and reverse input and concatanates the results along the feature axis.

        Args:
            input (torch.Tensor of shape (batch_size, sequence_length, input_size)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor of shape (n_samples, 2 * RNN.output_shape[0]) or (n_samples, sequence_length, 2 * RNN.output_shape[0]): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[-len(self.input_shape):] != self.input_shape:
            raise ValueError(f"Input shape {input.shape[-len(self.input_shape):]} does not match the expected shape {self.input_shape}.")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        forward_val = self.forward_layer.forward(input, training=training)
        backward_val = self.backward_layer.forward(input.flip(1), training=training)
        if not self.backward_layer.return_last:
            backward_val = backward_val.flip(1)
        self.output = torch.cat((forward_val, backward_val), dim=-1)
        return self.output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.

        Returns:
            torch.Tensor of shape (n_samples, sequence_length, input_size): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        forward_grad = dCdy[..., :self.output_shape[0] // 2]
        backward_grad = dCdy[..., self.output_shape[0] // 2:]  # self.forward_layer.output_shape[0]
        dCdx_forward = self.forward_layer.backward(forward_grad, **kwargs)
        dCdx_backward = self.backward_layer.backward(backward_grad.flip(1), **kwargs)
        if not self.backward_layer.return_last:
            dCdx_backward = dCdx_backward.flip(1)

        dCdx = dCdx_backward + dCdx_forward
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (*self.forward_layer.get_parameters(), *self.backward_layer.get_parameters())

    def get_nparams(self):
        return self.forward_layer.get_nparams() + self.backward_layer.get_nparams()
    
    def summary(self, offset=""):
        if not hasattr(self, "input_shape"):
            raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

        input_shape = "(seq_len, " + str(self.input_shape[0]) + ")"
        output_shape = str(self.output_shape[0]) if self.forward_layer.return_last else "(seq_len, " + str(self.output_shape[0]) + ")"
        super_summary = offset + f"{self.name} - (Input, Output): ({input_shape}, {output_shape})"
        sublayer_offset = offset + "    "
        forward_summary = "\n" + offset + self.forward_layer.summary(sublayer_offset)
        backward_summary = "\n" + offset + self.backward_layer.summary(sublayer_offset)
        return super_summary + forward_summary + backward_summary
