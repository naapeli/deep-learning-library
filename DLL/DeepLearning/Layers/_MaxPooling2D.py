import torch

from .Activations._Activation import Activation


"""
Pooling layer

input.shape = (batch_size, depth, input_height, input_width)
output.shape = (batch_size, depth, input_height // self.kernel_size, input_width // self.kernel_size)
"""
class MaxPooling2D(Activation):
    """
    The max pooling layer for a neural network.

    Args:
        pool_size (int): The pooling size used for the model. The pooling kernel is automatically square. Must be a positive integer.
    """
    def __init__(self, pool_size, **kwargs):
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError(f"pool_size must be a positive integer. Currently {pool_size}.")

        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.name = "MaxPooling2D"

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if not isinstance(input_shape, tuple | list) or len(input_shape) != 3:
            raise ValueError("input_shape must be a tuple of length 3.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        super().initialise_layer(input_shape, data_type, device)
        input_depth, input_height, input_width = self.input_shape
        self.output_shape = (input_depth, input_height // self.pool_size, input_width // self.pool_size)
    
    def _windows(self, input):
        # crop the rows and columns that do not fill a whole window and split the rest into windows
        batch_size, depth, height, width = input.shape
        out_height, out_width = height // self.pool_size, width // self.pool_size
        cropped = input[:, :, :out_height * self.pool_size, :out_width * self.pool_size]
        return cropped.reshape(batch_size, depth, out_height, self.pool_size, out_width, self.pool_size)

    def forward(self, input, **kwargs):
        """
        Applies the max pooling transformation.

        Args:
            input (torch.Tensor of shape (n_samples, input_depth, height, width)): The input to the layer. Must be a torch.Tensor of the spesified shape.
            
        Returns:
            torch.Tensor of shape (n_samples, output_depth, height // layer.pool_size, width // layer.pool_size): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")

        self.input = input
        self.output = self._windows(input).amax(dim=(3, 5))
        return self.output
    
    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer.

        Args:
            dCdy (torch.Tensor of shape (n_samples, output_depth, output_height, output_width) : The gradient given by the next layer.

        Returns:
            torch.Tensor of shape (n_samples, input_depth, input_height, input_width): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        # the gradient goes to the maximum of every window and is shared equally between ties
        windows = self._windows(self.input)
        is_max = (windows == self.output[:, :, :, None, :, None]).to(dCdy.dtype)
        counts = is_max.sum(dim=(3, 5), keepdim=True)
        window_gradient = is_max * dCdy[:, :, :, None, :, None] / counts
        batch_size, depth, height, width = self.input.shape
        dCdx = torch.zeros_like(self.input, dtype=dCdy.dtype)
        out_height, out_width = height // self.pool_size, width // self.pool_size
        dCdx[:, :, :out_height * self.pool_size, :out_width * self.pool_size] = window_gradient.reshape(batch_size, depth, out_height * self.pool_size, out_width * self.pool_size)
        return dCdx
