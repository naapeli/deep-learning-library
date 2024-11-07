import torch
from math import sqrt

from .BaseLayer import BaseLayer
from .Activations.Activation import Activation
from .Regularisation.BaseRegularisation import BaseRegularisation


class Dense(BaseLayer):
    """
    The basic dense linear layer.

    Args:
        output_shape (int): The output_shape of the model not containing the batch_size dimension. Must be a non-negative int. If is zero, the returned tensor is of shape (n_samples,) and if positive, the returned tensor is of shape (n_samples, output_shape).
        initialiser (str, optional): The initialisation method for models weights. Xavier should be used for tanh, sigmoid, softmax or other activations, which are approximately linear close to origin, while He should be used for the ReLU activation. Must be one of "Xavier_norm", "Xavier_uniform", "He_norm" or "He_uniform". Defaults to "Xavier_uniform".
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used fter this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, output_shape, initialiser="Xavier_uniform", activation=None, normalisation=None, **kwargs):
        if not isinstance(output_shape, int) or output_shape < 0:
            raise ValueError("output_shape must be a non-negative integer.")
        if initialiser not in ["Xavier_norm", "Xavier_uniform", "He_norm", "He_uniform"]:
            raise ValueError('initialiser must be one of "Xavier_norm", "Xavier_uniform", "He_norm" or "He_uniform".')
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__((output_shape,), activation=activation, normalisation=normalisation, **kwargs)
        self.name = "Dense"
        self.initialiser = initialiser

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if not isinstance(input_shape, tuple | list) or len(input_shape) != 1:
            raise ValueError("input_shape must be a tuple of length 1.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        super().initialise_layer(input_shape, data_type, device)
        
        input_dim = input_shape[0]
        output_dim = self.output_shape[0] if self.output_shape[0] > 1 else 1
        if self.initialiser == "Xavier_norm":
            self.weights = torch.normal(mean=0, std=sqrt(2/(input_dim + output_dim)), size=(input_dim, output_dim), dtype=self.data_type, device=self.device)
        elif self.initialiser == "Xavier_uniform":
            a = sqrt(6/(input_dim + output_dim))
            self.weights = 2 * a * torch.rand(size=(input_dim, output_dim), dtype=self.data_type, device=self.device) - a
        elif self.initialiser == "He_norm":
            self.weights = torch.normal(mean=0, std=sqrt(6/(input_dim)), size=(input_dim, output_dim), dtype=self.data_type, device=self.device)
        elif self.initialiser == "He_uniform":
            a = sqrt(12/(input_dim + output_dim))  # sqrt(6/input_dim)
            self.weights = 2 * a * torch.rand(size=(input_dim, output_dim), dtype=self.data_type, device=self.device) - a

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
            torch.Tensor of shape (n_samples,) if layer.output_shape[0] == 0 else (n_samples, layer.output_shape[0]): The output tensor after the transformations with the spesified shape.
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
        if self.output_shape[0] == 0: output = output.squeeze(dim=1)  # If the output_shape is a 1d tensor, remove the last dimension
        return output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of shape (n_samples,) if layer.output_shape[0] == 0 else (n_samples, layer.output_shape[0])): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, layer.input_shape[0]): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape and not (dCdy.ndim == 1 and self.output_shape[0] == 0):
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")
        
        if self.output_shape[0] == 0: dCdy = dCdy.unsqueeze(dim=1)  # If the output shape was a 1d tensor, add an extra dimension to the end
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        dCdx = dCdy @ self.weights.T
        self.weights.grad = self.input.T @ dCdy
        self.biases.grad = torch.mean(dCdy, axis=0)
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (self.weights, self.biases)
