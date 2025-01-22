import torch

from ._BaseLayer import BaseLayer
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser


class Dense(BaseLayer):
    """
    The basic dense linear layer.

    Args:
        output_shape (tuple[int] or int): The output_shape of the model not containing the batch_size dimension. Must contain non-negative integers. If is int, returned shape is (n_samples, int). If is the length is zero, the returned tensor is of shape (n_samples,). Otherwise the returned tensor is of shape (n_samples, *output_shape).
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, output_shape, bias=True, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
        if not isinstance(initialiser, Initialiser):
            raise ValueError('initialiser must be an instance of DLL.DeepLearning.Initialisers')
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "Dense"
        self.initialiser = initialiser
        self.bias = bias

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        super().initialise_layer(input_shape, data_type, device)
        
        input_dim = input_shape[-1]
        output_dim = 1 if len(self.output_shape) == 0 else self.output_shape[-1]
        
        self.weights = self.initialiser.initialise((input_dim, output_dim), data_type=self.data_type, device=self.device)
        self.biases = torch.zeros(output_dim, dtype=self.data_type, device=self.device)
        self.nparams = output_dim * input_dim + output_dim

    def forward(self, input, training=False, **kwargs):
        """
        Applies the basic linear transformation

        .. math::
        
            \\begin{align*}
                y_{lin} = xW + b,\\\\
                y_{reg} = f(y_{lin}),\\\\
                y_{activ} = g(y_{reg}),
            \\end{align*}
        
        where :math:`f` is the possible regularisation function and :math:`g` is the possible activation function.

        Args:
            input (torch.Tensor of shape (n_samples, n_features)): The input to the dense layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor of shape (n_samples,) if len(layer.output_shape) == 0 else (n_samples, output_shape): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        self.input = input
        output = self.input @ self.weights + self.biases
        if self.normalisation: output = self.normalisation.forward(output, training=training)
        if self.activation: output = self.activation.forward(output)
        if len(self.output_shape) == 0: output = output.squeeze(dim=1)  # If the output_shape is a 1d tensor, remove the last dimension
        return output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of shape (n_samples,) if len(layer.output_shape) == 0 else (n_samples, output_shape)): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, layer.input_shape[0]): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")
        
        if len(self.output_shape) == 0: dCdy = dCdy.unsqueeze(dim=1)  # If the output shape was a 1d tensor, add an extra dimension to the end
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        dCdx = dCdy @ self.weights.T
        dCdW = self.input.transpose(-1, -2) @ dCdy
        self.weights.grad = torch.mean(dCdW, dim=tuple(range(self.input.ndim - 2))) if self.input.ndim > 2 else dCdW
        self.biases.grad = torch.mean(dCdy, dim=tuple(range(self.input.ndim - 1))) if self.bias else torch.zeros_like(self.biases)
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (self.weights, self.biases)
