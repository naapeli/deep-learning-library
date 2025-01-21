import torch
import torch.nn.functional as F
import numpy as np

from ._BaseLayer import BaseLayer
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser


class Conv2D(BaseLayer):
    """
    The convolutional layer for a neural network.

    Args:
        kernel_size (int): The kernel size used for the model. The kernel is automatically square. Must be a positive integer.
        output_depth (int): The output depth of the layer. Must be a positive integer.
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used fter this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, kernel_size, output_depth, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer. Currently {kernel_size}.")
        if not isinstance(output_depth, int) or output_depth <= 0:
            raise ValueError(f"output_depth must be a positive integer. Currently {output_depth}.")
        if not isinstance(initialiser, Initialiser):
            raise ValueError('initialiser must be an instance of DLL.DeepLearning.Initialisers')
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__(output_shape=None, activation=activation, normalisation=normalisation, **kwargs)
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.initialiser = initialiser
        self.name = "Conv2D"

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        input_depth, input_height, input_width = input_shape
        self.input_depth = input_depth
        self.output_shape = (self.output_depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)
        self.nparams = np.prod(self.kernels_shape) + np.prod(self.output_shape)
        
        super().initialise_layer(input_shape, data_type, device)

        self.kernels = self.initialiser.initialise(self.kernels_shape, data_type=self.data_type, device=self.device)
        self.biases = torch.zeros(self.output_shape)
        
        if self.activation:
            self.activation.initialise_layer(self.output_shape, self.data_type, self.device)
        if self.normalisation:
            self.normalisation.initialise_layer(self.output_shape, self.data_type, self.device)
    
    def forward(self, input, training=False, **kwargs):
        """
        Applies the convolutional transformation.

        .. math::
            \\begin{align*}
                y_{i, j} &= \\text{bias}_j + \\sum_{k = 1}^{\\text{d_in}} \\text{kernel}(j, k) \star \\text{input}(i, k),\\\\
                y_{reg_{i, j}} &= f(y_{i, j}),\\\\
                y_{activ_{i, j}} &= g(y_{reg}),
            \\end{align*}
        
        where :math:`\star` is the cross-correlation operator, :math:`\\text{d_in}` is the input_depth, :math:`i\in [1,\dots, \\text{batch_size}]`, :math:`j\in[1,\dots, \\text{output_depth}]`, :math:`f` is the possible regularisation function and :math:`g` is the possible activation function.

        Args:
            input (torch.Tensor of shape (n_samples, input_depth, input_height, input_width)): The input to the layer. Must be a torch.Tensor of the spesified shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor of shape (n_samples, output_depth, height - kernel_size + 1, width - kernel_size + 1): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        batch_size = input.shape[0]
        self.input = input
        self.output = self.biases.clone().repeat(batch_size, 1, 1, 1)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                conv_output = F.conv2d(input[:, j:j+1, :, :], self.kernels[i:i+1, j:j+1, :, :], padding="valid")
                self.output[:, i, :, :] += conv_output[:, 0, :, :]
                
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of shape (n_samples, output_depth, output_height, output_width) : The gradient given by the next layer.

        Returns:
            torch.Tensor of shape (n_samples, input_depth, input_height, input_width): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        kernel_gradient = torch.zeros_like(self.kernels, device=self.device, dtype=self.data_type)
        dCdx = torch.zeros_like(self.input, device=self.device, dtype=self.data_type)
        batch_size = self.input.shape[0]
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] = F.conv2d(self.input[:, j:j+1, :, :], dCdy[:, i:i+1, :, :], padding="valid")[0, 0, :, :]
                dCdx[:, j] += F.conv2d(dCdy[:, i:i+1, :, :], torch.flip(self.kernels[i:i+1, j:j+1, :, :], dims=(2, 3)), padding=[self.kernel_size - 1, self.kernel_size - 1])[0, 0, :, :]
                
        self.biases.grad = dCdy.mean(dim=0)
        self.kernels.grad = kernel_gradient / batch_size
        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (self.kernels, self.biases)
