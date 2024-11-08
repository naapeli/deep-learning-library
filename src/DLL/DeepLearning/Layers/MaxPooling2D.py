import torch

from .Activations.Activation import Activation


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

        super().__init__(output_shape=None, **kwargs)
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
    
    def generate_sections(self, image_batch):
        """
        :meta private:
        """
        if image_batch.ndim != 4:
            raise ValueError("image_batch should be 4 dimensional.")

        height, width = image_batch.shape[2] // self.pool_size, image_batch.shape[3] // self.pool_size
        for h in range(height):
            for w in range(width):
                slice = image_batch[:, :, (h * self.pool_size):(h * self.pool_size + self.pool_size), (w * self.pool_size):(w * self.pool_size + self.pool_size)]
                yield slice, h, w
    
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
        batch_size, depth, widht, height = self.input.shape
        self.output = torch.zeros(size=(batch_size, depth, widht // self.pool_size, height // self.pool_size), device=input.device, dtype=input.dtype)
        for slice, h, w in self.generate_sections(input):
            self.output[:, :, h, w] = torch.amax(slice, dim=(2, 3))
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

        dCdx = torch.zeros_like(self.input, device=dCdy.device, dtype=dCdy.dtype)
        sums = torch.ones_like(self.input, device=dCdy.device, dtype=dCdy.dtype)
        for slice, h, w in self.generate_sections(self.input):
            derivative_slice = dCdx[:, :, h * self.pool_size:h * self.pool_size + self.pool_size, w * self.pool_size:w * self.pool_size + self.pool_size]
            max_vals = self.output[:, :, h, w].unsqueeze(-1).unsqueeze(-1)
            selector = torch.eq(max_vals.repeat(1, 1, self.pool_size, self.pool_size), slice)
            sums[:, :, h * self.pool_size:h * self.pool_size + self.pool_size, w * self.pool_size:w * self.pool_size + self.pool_size] = torch.sum(selector, dim=(2, 3), keepdim=True).repeat(1, 1, self.pool_size, self.pool_size)
            derivatives = dCdy[:, :, h, w].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.pool_size, self.pool_size)
            dCdx[:, :, h * self.pool_size:h * self.pool_size + self.pool_size, w * self.pool_size:w * self.pool_size + self.pool_size] = torch.where(selector, derivatives, derivative_slice)
        return dCdx / sums
