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
        self.output_shape = (self.output_depth, input_height, input_width)
        
        pad_total = self.kernel_size - 1
        self.pad_beg = pad_total // 2
        self.pad_end = pad_total - self.pad_beg
        self.padding_tuple = (self.pad_beg, self.pad_end, self.pad_beg, self.pad_end)

        self.kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)
        self.nparams = np.prod(self.kernels_shape) + self.output_depth
        
        super().initialise_layer(input_shape, data_type, device)

        self.kernels = self.initialiser.initialise(self.kernels_shape, data_type=self.data_type, device=self.device)
        self.biases = torch.zeros((self.output_depth,), dtype=self.data_type, device=self.device)
        
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
            torch.Tensor of shape (n_samples, output_depth, input_height, input_width): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the specified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        self.input = input
        
        self.output = F.conv2d(self.input, self.kernels, bias=self.biases, padding="same")

        if self.normalisation: 
            self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: 
            self.output = self.activation.forward(self.output)
            
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
            raise ValueError(f"dCdy is not the same shape as the specified output_shape ({dCdy.shape[1:], self.output_shape}).")

        if self.activation: 
            dCdy = self.activation.backward(dCdy)
        if self.normalisation: 
            dCdy = self.normalisation.backward(dCdy)
        
        self.biases.grad += dCdy.sum(dim=(0, 2, 3))

        input_padded = F.pad(self.input, self.padding_tuple)
        
        input_permuted = input_padded.transpose(0, 1)
        grad_permuted = dCdy.transpose(0, 1)

        grad_kernels = F.conv2d(input_permuted, grad_permuted)
        
        self.kernels.grad += grad_kernels.transpose(0, 1)

        dCdx_full = F.conv_transpose2d(dCdy, self.kernels, padding=0)
        
        height, width = self.input.shape[2], self.input.shape[3]
        dCdx = dCdx_full[:, :, self.pad_beg : self.pad_beg + height, self.pad_beg : self.pad_beg + width]

        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (self.kernels, self.biases, *super().get_parameters())
