import torch

from ._BaseLayer import BaseLayer
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser


def _make_knots(n_splines, degree, bounds, data_type=torch.float32, device=torch.device("cpu")):
    """
    Uniform knot vector for ``n_splines`` B-splines of the given degree. The knots are extended by ``degree``
    steps past both bounds, so the splines form a partition of unity on the whole interval ``bounds``.
    """
    n_intervals = n_splines - degree  # number of knot intervals inside bounds
    step = (bounds[1] - bounds[0]) / n_intervals
    knots = bounds[0] + step * torch.arange(-degree, n_splines + 1, dtype=data_type, device=device)
    return knots  # (n_splines + degree + 1,)


def _SiLU(x):
    return x * torch.sigmoid(x)


def _SiLU_der(x):
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1 + x * (1 - sigmoid))


def _evaluate_basis(x, knots, degree, derivative=False):
    """
    Evaluates the SiLU function and every B-spline basis function at once using the Cox-de Boor recursion.

    Args:
        x (torch.Tensor of shape (...)): The evaluation points.
        knots (torch.Tensor of shape (n_splines + degree + 1,) or None): The knot vector. If None, only SiLU is used.
        degree (int): The degree of the B-splines.
        derivative (bool, optional): If True, also returns the derivatives of the basis functions. Defaults to False.

    Returns:
        torch.Tensor of shape (..., n_splines + 1), and optionally the derivatives of the same shape. Index 0 is SiLU.
    """
    values = [_SiLU(x).unsqueeze(-1)]
    derivatives = [_SiLU_der(x).unsqueeze(-1)] if derivative else None

    if knots is not None:
        x = x.unsqueeze(-1)
        splines = ((knots[:-1] <= x) & (x < knots[1:])).to(x.dtype)  # degree 0 splines (..., n_knots - 1)
        for k in range(1, degree + 1):
            previous = splines
            left_denom = knots[k:-1] - knots[:-k - 1]  # the knots are strictly increasing, so never zero
            right_denom = knots[k + 1:] - knots[1:-k]
            splines = (x - knots[:-k - 1]) / left_denom * previous[..., :-1] + (knots[k + 1:] - x) / right_denom * previous[..., 1:]
        values.append(splines)
        if derivative:
            derivatives.append(degree * (previous[..., :-1] / left_denom - previous[..., 1:] / right_denom))

    values = torch.cat(values, dim=-1)
    if derivative:
        return values, torch.cat(derivatives, dim=-1)
    return values


class DenseKAN(BaseLayer):
    """
    The dense Kolmogorov-Arnold network layer. The implementation is based on `this paper <https://arxiv.org/pdf/2404.19756>`_ and `this article <https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/>`_.

    Args:
        output_shape (tuple[int] or int): The output_shape of the model not containing the batch_size dimension. Must contain non-negative integers. If is int, returned shape is (n_samples, int). If is the length is zero, the returned tensor is of shape (n_samples,). Otherwise the returned tensor is of shape (n_samples, *output_shape).
        n_basis_funcs (int, optional): The number of basis functions used for fitting. If 1, only SiLU is used. Otherwise 1 SiLU and n_basis_funcs - 1 Bsplines basis functions are used. Must be 1 or an integer of at least basis_func_degree + 2. Defaults to 10.
        bounds (tuple[int | float], optional): The theoretical min and max of the data that will be passed to the forward method. The B-splines form a partition of unity on this interval. Must be a tuple containing two numbers. Defaults to (-1, 1).
        basis_func_degree (int, optional): The degree of the Bspline basis functions. Must be positive. Defaults to 3.
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    Note:
        All edge functions of the layer are stored in a single weight tensor of shape (output_dim, input_dim, n_basis_funcs) and the basis functions are evaluated only once per forward pass, so the whole layer is computed with a single matrix multiplication.
    """
    def __init__(self, output_shape, n_basis_funcs=10, bounds=(-1, 1), basis_func_degree=3, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
        if not isinstance(n_basis_funcs, int) or n_basis_funcs <= 0:
            raise ValueError("n_basis_funcs must be a positive integer.")
        if not isinstance(bounds, tuple) or len(bounds) != 2 or not all(isinstance(bound, (int, float)) for bound in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("bounds should be a tuple of length 2 containing the theoretical minimum and maximum of the data that will be passed in.")
        if not isinstance(basis_func_degree, int) or basis_func_degree <= 0:
            raise ValueError("basis_func_degree must be a positive integer.")
        if n_basis_funcs != 1 and n_basis_funcs < basis_func_degree + 2:
            raise ValueError("n_basis_funcs must be 1 (only SiLU) or at least basis_func_degree + 2.")
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
        self.bounds = bounds
        self.basis_func_degree = basis_func_degree

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        super().initialise_layer(input_shape, data_type, device)
        
        if len(input_shape) != 1:
            raise ValueError("input_shape must be of the form (n_features,).")

        input_dim = input_shape[0]
        output_dim = 1 if len(self.output_shape) == 0 else self.output_shape[-1]

        n_splines = self.n_basis_funcs - 1
        self.knots = _make_knots(n_splines, self.basis_func_degree, self.bounds, data_type=data_type, device=device) if n_splines > 0 else None
        self.weights = self.initialiser.initialise((output_dim, input_dim, self.n_basis_funcs), data_type=data_type, device=device)
        self.nparams = self.weights.numel()

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
        if training:
            self.basis_values, self.basis_derivatives = _evaluate_basis(input, self.knots, self.basis_func_degree, derivative=True)
        else:
            self.basis_values, self.basis_derivatives = _evaluate_basis(input, self.knots, self.basis_func_degree), None
        # (n, input_dim * n_basis_funcs) @ (input_dim * n_basis_funcs, output_dim)
        output = self.basis_values.flatten(start_dim=1) @ self.weights.flatten(start_dim=1).T
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
        if self.basis_derivatives is None:
            self.basis_values, self.basis_derivatives = _evaluate_basis(self.input, self.knots, self.basis_func_degree, derivative=True)

        self.weights.grad += (dCdy.T @ self.basis_values.flatten(start_dim=1)).view_as(self.weights)
        dCdbasis = (dCdy @ self.weights.flatten(start_dim=1)).view_as(self.basis_values)  # (n, input_dim, n_basis_funcs)
        dCdx = (dCdbasis * self.basis_derivatives).sum(dim=-1)
        return dCdx

    def get_parameters(self):
        """
        :meta private:
        """
        return (self.weights, *super().get_parameters())
