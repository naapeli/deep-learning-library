import torch
import torch.nn.functional as F
from math import sqrt, ceil
import numpy as np

from .BaseLayer import BaseLayer


"""
Convolutional layer

input.shape = (batch_size, input_depth, input_height, input_widht)
output.shape = (batch_size, output_depth, input_height - kernel_size + 1, input_widht - kernel_size + 1)
"""
class Conv2D(BaseLayer):
    def __init__(self, kernel_size, output_depth, input_shape=None, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape=None, activation=activation, normalisation=normalisation, **kwargs)
        self.input_shape = input_shape
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.name = "Conv2D"
        if input_shape: self.initialise_layer()

    def initialise_layer(self):
        batch_size, input_depth, input_height, input_width = self.input_shape
        self.input_depth = input_depth
        self.output_shape = (batch_size, self.output_depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)
        self.kernels = 1. / sqrt(input_depth) * (2 * torch.rand(size=self.kernels_shape, device=self.device, dtype=self.data_type) - 1)
        self.biases = 1. / sqrt(input_depth) * (2 * torch.rand(size=self.output_shape[1:], device=self.device, dtype=self.data_type) - 1)
        self.nparams = np.prod(self.kernels_shape) + np.prod(self.output_shape[1:])
        if self.activation:
            self.activation.set_output_shape(self.output_shape)
        if self.normalisation:
            self.normalisation.data_type = self.data_type
            self.normalisation.device = self.device
            self.normalisation.set_output_shape(self.output_shape)
    
    def forward(self, input, training=False, **kwargs):
        batch_size = input.shape[0]
        self.input = input
        self.output = self.biases.clone().repeat(batch_size, 1, 1, 1)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                conv_output = F.conv2d(input[:, j:j+1, :, :], self.kernels[i:i+1, j:j+1, :, :], padding="valid")
                self.output[:, i, :, :] += conv_output[:, 0, :, :]
                
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output
    
    def backward(self, dCdy, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        kernel_gradient = torch.zeros_like(self.kernels, device=self.device, dtype=self.data_type)
        dCdx = torch.zeros_like(self.input, device=self.device, dtype=self.data_type)
        batch_size = self.input.shape[0]
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] = F.conv2d(self.input[:, j:j+1, :, :], dCdy[:, i:i+1, :, :], padding="valid")[0, 0, :, :]
                dCdx[:, j] += F.conv2d(dCdy[:, i:i+1, :, :], torch.flip(self.kernels[i:i+1, j:j+1, :, :], dims=(2, 3)), padding=[self.kernel_size - 1, self.kernel_size - 1])[0, 0, :, :]
                
        self.biases.grad = dCdy.mean(dim=0)
        self.kernels.grad = kernel_gradient / batch_size
        return dCdx
    
    def get_parameters(self):
        return (self.kernels, self.biases)
