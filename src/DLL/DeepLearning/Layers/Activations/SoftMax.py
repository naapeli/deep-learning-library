import torch
from .Activation import Activation


class SoftMax(Activation):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, **kwargs)
        self.name = "Softmax"

    def forward(self, input, **kwargs):
        self.input = input
        exponential_input = torch.exp(self.input - torch.max(self.input, dim=1, keepdim=True).values)
        self.output = exponential_input / torch.sum(exponential_input, dim=1, keepdim=True)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        n = self.output_shape[1]
        # dCdx = torch.stack([(dCdy[i] @ (torch.tile(datapoint, (n, 1)).T * (torch.eye(n, device=self.device, dtype=self.data_type) - torch.tile(datapoint, (n, 1))))) for i, datapoint in enumerate(self.output)])

        # same calculations as above, but faster
        datapoints_expanded = self.output.unsqueeze(1).repeat(1, n, 1)
        identity_matrix = torch.eye(n, device=dCdy.device, dtype=dCdy.dtype)
        matrix_diff = identity_matrix - datapoints_expanded
        dCdx = dCdy.unsqueeze(1) @ (datapoints_expanded.transpose(1, 2) * matrix_diff)
        return dCdx.squeeze(1)
