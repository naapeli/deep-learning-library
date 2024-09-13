import torch

from .Activations.Activation import Activation


"""
Pooling layer

input.shape = (batch_size, depth, input_height, input_width)
output.shape = (batch_size, depth, input_height // self.kernel_size, input_width // self.kernel_size)
"""
class MaxPooling2D(Activation):
    def __init__(self, pool_size, input_shape=None, **kwargs):
        super().__init__(output_shape=None, **kwargs)
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.name = "MaxPooling2D"
        if input_shape: self.initialise_layer()

    def initialise_layer(self):
        batch_size, input_depth, input_height, input_width = self.input_shape
        self.output_shape = (batch_size, input_depth, input_height // self.pool_size, input_width // self.pool_size)
    
    def generate_sections(self, image_batch):
        height, width = image_batch.shape[2] // self.pool_size, image_batch.shape[3] // self.pool_size
        for h in range(height):
            for w in range(width):
                slice = image_batch[:, :, (h * self.pool_size):(h * self.pool_size + self.pool_size), (w * self.pool_size):(w * self.pool_size + self.pool_size)]
                yield slice, h, w
    
    def forward(self, input, **kwargs):
        self.input = input
        batch_size, depth, widht, height = self.input.shape
        self.output = torch.zeros(size=(batch_size, depth, widht // self.pool_size, height // self.pool_size), device=input.device, dtype=input.dtype)
        for slice, h, w in self.generate_sections(input):
            self.output[:, :, h, w] = torch.amax(slice, dim=(2, 3))
        return self.output
    
    def backward(self, dCdy, **kwargs):
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
