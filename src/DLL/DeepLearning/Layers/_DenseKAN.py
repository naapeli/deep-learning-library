import torch
from functools import lru_cache, partial

from ._BaseLayer import BaseLayer
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser


@lru_cache
def _bspline_basis(i, n_fun, knots, x):
    if n_fun == 0:
        return ((knots[i] <= x) & (x < knots[i + 1])).float()
    
    denom1 = knots[i + n_fun] - knots[i]
    denom2 = knots[i + n_fun + 1] - knots[i + 1]

    term1 = (x - knots[i]) / denom1 * _bspline_basis(i, n_fun - 1, knots, x) if denom1 > 0 else torch.zeros_like(x)
    term2 = (knots[i + n_fun + 1] - x) / denom2 * _bspline_basis(i + 1, n_fun - 1, knots, x) if denom2 > 0 else torch.zeros_like(x)
    return term1 + term2

@lru_cache
def _bspline_basis_derivative(i, n_fun, knots, x):
    if n_fun == 0:
        return torch.zeros(x)

    denom1 = knots[i + n_fun] - knots[i]
    denom2 = knots[i + n_fun + 1] - knots[i + 1]

    term1 = _bspline_basis(i, n_fun - 1, knots, x) / denom1 if denom1 > 0 else torch.zeros_like(x)
    term2 = _bspline_basis(i + 1, n_fun - 1, knots, x) / denom2 if denom2 > 0 else torch.zeros_like(x)

    return n_fun * (term1 - term2)

def _SiLU(x):
    return x / (1 + torch.exp(-x))

def _SiLU_der(x):
    return (1 + torch.exp(-x) + x * torch.exp(-x)) / (1 + torch.exp(-x)) ** 2

def make_basis_func(i, degree, t, x):
    return _bspline_basis(i, degree, t, x)

def make_basis_func_der(i, degree, t, x):
    return _bspline_basis_derivative(i, degree, t, x)

def _get_basis_functions(n_fun, degree=3, bounds=(-1, 1)):
    grid_len = n_fun - degree + 1
    step = (bounds[1] - bounds[0]) / (grid_len - 1)
    edge_funcs, edge_func_ders = [], []

    # SiLU bias function
    edge_funcs.append(_SiLU)
    edge_func_ders.append(_SiLU_der)

    # B-splines
    t = torch.linspace(bounds[0] - degree * step, bounds[1] + degree * step, grid_len + 2 * degree)
    t[degree], t[-degree - 1] = bounds[0], bounds[1]
    for ind_spline in range(n_fun - 1):
        edge_funcs.append(partial(make_basis_func, ind_spline, degree, t))
        edge_func_ders.append(partial(make_basis_func_der, ind_spline, degree, t))
    return edge_funcs, edge_func_ders


class _NeuronKAN:
    def __init__(self, input_length, n_basis_funcs, bounds, initialiser, data_type, device, basis_func_degree=3):
        self.edge_fun, self.edge_fun_der = _get_basis_functions(n_fun=n_basis_funcs, degree=basis_func_degree, bounds=bounds)
        self.weights = initialiser.initialise((n_basis_funcs, input_length), data_type=data_type, device=device)

    def forward(self, input, **kwargs):
        self.input = input
        self.edge_func_values = torch.stack([func(input) for func in self.edge_fun], dim=1)  # (n, n_basis_funcs, input_length)
        output = (self.weights.unsqueeze(0) * self.edge_func_values).sum(dim=(1, 2))
        return output

    def backward(self, dCdy, **kwargs):
        dCdy = dCdy.view(len(dCdy), 1, 1)  # (n, 1, 1)
        self.weights.grad += torch.sum(dCdy * self.edge_func_values, dim=0)
        dCdx = dCdy * self.weights.unsqueeze(0)  # (n, n_basis_funcs, input_length)
        edge_func_derivatives = torch.stack([func(self.input) for func in self.edge_fun_der], dim=1)
        dCdx = torch.sum(dCdx * edge_func_derivatives, dim=1)
        return dCdx


class DenseKAN(BaseLayer):
    """
    The dense Kolmogorov-Arnold network layer. The implementation is based on `this paper <https://arxiv.org/pdf/2404.19756>`_ and `this article <https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/>`_.

    Args:
        output_shape (tuple[int] or int): The output_shape of the model not containing the batch_size dimension. Must contain non-negative integers. If is int, returned shape is (n_samples, int). If is the length is zero, the returned tensor is of shape (n_samples,). Otherwise the returned tensor is of shape (n_samples, *output_shape).
        n_basis_funcs (int, optional): The number of basis functions used for fitting. If 1, only SiLU is used. Otherwise 1 SiLU and n_basis_funcs - 1 Bsplines basis functions are used. Must be a positive integer. Defaults to 10.
        bounds (tuple[int], optional): The theoretical min and max of the data that will be passed to the forward method. Must be a tuple containing two integers. Defaults to (-1, 1).
        basis_func_degree (int, optional): The degree of the Bspline basis functions. Must be positive. Defaults to 3.
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    Note:
        n_basis_funcs and basis_func_degree should be different to avoid certain errors.
    """
    def __init__(self, output_shape, n_basis_funcs=10, bounds=(-1, 1), basis_func_degree=3, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
        if not isinstance(n_basis_funcs, int) or n_basis_funcs <= 0:
            raise ValueError("n_basis_funcs must be a positive integer.")
        if not isinstance(bounds, tuple) or len(bounds) != 2 or not isinstance(bounds[0], int) or not isinstance(bounds[1], int) or bounds[0] >= bounds[1]:
            raise ValueError("bounds should be a tuple of length 2 containing the theoretical minimum and maximum of the data that will be passed in.")
        if not isinstance(basis_func_degree, int) or basis_func_degree <= 0:
            raise ValueError("basis_func_degree must be a positive integer.")
        if n_basis_funcs == basis_func_degree:
            raise ValueError("n_basis_funcs and basis_func_degree should be different.")
        if not isinstance(initialiser, Initialiser):
            raise ValueError('initialiser must be an instance of DLL.DeepLearning.Initialisers')
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "DenseKAN"
        self.initialiser = initialiser
        self.n_basis_funcs = n_basis_funcs

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        super().initialise_layer(input_shape, data_type, device)
        
        input_dim = input_shape[-1]
        output_dim = 1 if len(self.output_shape) == 0 else self.output_shape[-1]
        
        self.neurons = [_NeuronKAN(input_dim, self.n_basis_funcs, (-1, 1), self.initialiser, data_type, device, 3) for _ in range(output_dim)]  # TODO: Change the hard coded values to actual parameters
        self.nparams = self.n_basis_funcs * output_dim * input_dim

    def forward(self, input, training=False, **kwargs):
        """
        Applies the forward equation of the Kolmogorov-Arnold network.

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
        output = torch.stack([neuron.forward(input) for neuron in self.neurons], dim=1)
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
        dCdx = sum([neuron.backward(dCdy[:, i]) for i, neuron in enumerate(self.neurons)])
        return dCdx

    def get_parameters(self):
        """
        :meta private:
        """
        return (*(neuron.weights for neuron in self.neurons), *super().get_parameters())
