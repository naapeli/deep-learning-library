import torch

from .Activations._Activation import Activation


class Upsampling2D(Activation):
    """
    The nearest-neighbour upsampling layer for a neural network. Every pixel is repeated scale_factor times in both spatial directions, so the height and the width of the input are multiplied by scale_factor. Together with :class:`Conv2D`, the layer can be used to build convolutional decoders and generators, which turn small feature maps into larger images.

    Args:
        scale_factor (int): The factor by which the height and the width are multiplied. Must be a positive integer.
    """
    def __init__(self, scale_factor, **kwargs):
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive integer. Currently {scale_factor}.")
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.name = "Upsampling2D"

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
        self.output_shape = (input_depth, input_height * self.scale_factor, input_width * self.scale_factor)

    def forward(self, input, **kwargs):
        """
        Repeats every pixel of the input scale_factor times in both spatial directions.

        Args:
            input (torch.Tensor of shape (n_samples, depth, height, width)): The input to the layer. Must be a torch.Tensor of the spesified shape.

        Returns:
            torch.Tensor of shape (n_samples, depth, height * layer.scale_factor, width * layer.scale_factor): The upsampled input.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")

        self.input = input
        self.output = input.repeat_interleave(self.scale_factor, dim=2).repeat_interleave(self.scale_factor, dim=3)
        return self.output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Every input pixel receives the sum of the gradients of its copies.

        Args:
            dCdy (torch.Tensor of shape (n_samples, depth, height * layer.scale_factor, width * layer.scale_factor)): The gradient given by the next layer.

        Returns:
            torch.Tensor of shape (n_samples, depth, height, width): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        batch_size, depth, height, width = self.input.shape
        s = self.scale_factor
        return dCdy.reshape(batch_size, depth, height, s, width, s).sum(dim=(3, 5))
