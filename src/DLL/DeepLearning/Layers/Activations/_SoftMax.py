import torch

from ._Activation import Activation


class SoftMax(Activation):
    """
    The softmax activation function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Softmax"

    def forward(self, input, **kwargs):
        """
        Calculates the following function for every element of the input matrix:

        .. math::
        
            \\text{Softmax}(x)_i = \\frac{e^{x_i}}{\\sum_{j=1}^{K} e^{x_j}},
        
        where :math:`K` is the number of features of the input.

        Args:
            input (torch.Tensor of shape (n_samples, n_features)): The input to the layer. Must be a torch.Tensor of the spesified shape.

        Returns:
            torch.Tensor: The output tensor after applying the activation function of the same shape as the input.
        """
        if not isinstance(input, torch.Tensor) or input.ndim != 2:
            raise TypeError("input must be a 2 dimensional torch.Tensor.")

        self.input = input
        exponential_input = torch.exp(self.input - torch.max(self.input, dim=1, keepdim=True).values)
        self.output = exponential_input / torch.sum(exponential_input, dim=1, keepdim=True)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, n_features): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape != self.output.shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        n = dCdy.shape[1]
        # dCdx = torch.stack([(dCdy[i] @ (torch.tile(datapoint, (n, 1)).T * (torch.eye(n, device=self.device, dtype=self.data_type) - torch.tile(datapoint, (n, 1))))) for i, datapoint in enumerate(self.output)])

        # same calculations as above, but faster
        datapoints_expanded = self.output.unsqueeze(1).repeat(1, n, 1)
        identity_matrix = torch.eye(n, device=dCdy.device, dtype=dCdy.dtype)
        matrix_diff = identity_matrix - datapoints_expanded
        dCdx = dCdy.unsqueeze(1) @ (datapoints_expanded.transpose(1, 2) * matrix_diff)
        return dCdx.squeeze(1)
