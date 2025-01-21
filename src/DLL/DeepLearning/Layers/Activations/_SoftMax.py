import torch

from ._Activation import Activation


class SoftMax(Activation):
    """
    The softmax activation function.

    Args:
        dim (int): The dimension on which the softmax is calculated. If the data is (n_samples, n_channels, n_features) and one wants to calculate the softmax on the channels, one should select dim=1 or dim=-2.
    """
    def __init__(self, dim=-1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
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
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.ndim <= self.dim or -input.ndim > self.dim:
            raise ValueError(f"dim must be in range [{-input.ndim}, {input.ndim}) for input of this shape, but is currently {self.dim}.")

        self.input = input
        exponential_input = torch.exp(self.input - torch.max(self.input, dim=self.dim, keepdim=True).values)
        self.output = exponential_input / torch.sum(exponential_input, dim=self.dim, keepdim=True)
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

        # v1 - 2 dimensional
        # n = dCdy.shape[1]
        # # dCdx = torch.stack([(dCdy[i] @ (torch.tile(datapoint, (n, 1)).T * (torch.eye(n, device=self.device, dtype=self.data_type) - torch.tile(datapoint, (n, 1))))) for i, datapoint in enumerate(self.output)])

        # v2 - 2 dimensional, but faster
        # datapoints_expanded = self.output.unsqueeze(1).repeat(1, n, 1)
        # identity_matrix = torch.eye(n, device=dCdy.device, dtype=dCdy.dtype)
        # matrix_diff = identity_matrix - datapoints_expanded
        # dCdx = dCdy.unsqueeze(1) @ (datapoints_expanded.transpose(1, 2) * matrix_diff)
        # return dCdx.squeeze(1)

        # v3 - n dimensional
        # dim = self.dim if self.dim >= 0 else dCdy.ndim + self.dim
        # n = self.output.shape[dim]
        # output_expanded = self.output.unsqueeze(dim)
        # identity_matrix = torch.eye(n, device=dCdy.device, dtype=dCdy.dtype).reshape((1,) * dim + (n, n) + (1,) * (dCdy.ndim - dim - 1))
        # matrix_diff = identity_matrix - output_expanded
        # # probably not optimal permuting the two wanted dimensions to the end for the matrix multiplication
        # dCdx = dCdy.unsqueeze(dim).permute(*range(dim), *range(dim + 2, dCdy.ndim + 1), dim, dim + 1) @ (output_expanded.transpose(dim, dim + 1) * matrix_diff).permute(*range(dim), *range(dim + 2, dCdy.ndim + 1), dim, dim + 1)
        # return dCdx.permute(*range(dim), dCdy.ndim - 1, dCdy.ndim, *range(dim, dCdy.ndim - 1)).squeeze(dim)

        # v4 - n dimensional, but faster
        output = self.output
        dCdx = output * (dCdy - torch.sum(dCdy * output, dim=self.dim, keepdim=True))
        return dCdx
