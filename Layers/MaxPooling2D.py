import torch
from Layers.Activations.Activation import Activation


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
        self.output = torch.zeros(size=(batch_size, depth, widht // self.pool_size, height // self.pool_size), device=self.device, dtype=self.data_type)
        for slice, h, w in self.generate_sections(input):
            self.output[:, :, h, w] = torch.amax(slice, dim=(2, 3))
        return self.output
    
    def backward(self, dCdy, **kwargs):
        dCdx = torch.zeros_like(self.input, device=self.device, dtype=self.data_type)
        for slice, h, w in self.generate_sections(self.input):
            batch_size, depth, h0, w0 = slice.shape
            max_vals = torch.amax(slice, dim=(2, 3), keepdim=True)
            sums = torch.sum(torch.eq(max_vals.repeat(1, 1, self.pool_size, self.pool_size), slice), dim=(2, 3), keepdim=True)
            max_vals = max_vals
            for batch in range(batch_size):
                for idx_h in range(h0):
                    for idx_w in range(w0):
                        for idx_k in range(depth):
                            if slice[batch, idx_k, idx_h, idx_w] == max_vals[batch, idx_k, 0, 0]:
                                dCdx[batch, idx_k, h * self.pool_size + idx_h, w * self.pool_size + idx_w] = dCdy[batch, idx_k, h, w]
        return dCdx / sums
