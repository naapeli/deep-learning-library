import torch
import torch.nn.functional as F
from Layers.Base import Base
from math import sqrt
import numpy as np


"""
Convolutional layer

input.shape = (batch_size, input_depth, input_height, input_widht)
output.shape = (batch_size, output_depth, input_height - kernel_size + 1, input_widht - kernel_size + 1)
"""
class Conv2D(Base):
    def __init__(self, kernel_size, output_depth, input_shape=None, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape=None, activation=activation, normalisation=normalisation, **kwargs)
        self.input_shape = input_shape
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.name = "Conv2D"
        if input_shape: self.initialise_layer()

    def initialise_layer(self):
        input_depth, input_height, input_width = self.input_shape
        self.input_depth = input_depth
        self.output_shape = (self.output_depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)
        self.kernels = 1. / sqrt(input_depth) * (2 * torch.rand(size=self.kernels_shape, device=self.device, dtype=self.data_type) - 1)
        self.biases = 1. / sqrt(input_depth) * (2 * torch.rand(size=self.output_shape, device=self.device, dtype=self.data_type) - 1)
        self.nparams = np.prod(self.kernels_shape) + np.prod(self.output_shape)
        if self.activation:
            self.activation.set_output_shape(self.output_shape)
        if self.normalisation:
            self.normalisation.set_output_shape(self.output_shape)
    
    def forward(self, input, training=False, **kwargs):
        batch_size = input.shape[0]
        self.input = input
        self.output = torch.stack([self._forward_single(input[batch:batch+1]) for batch in range(batch_size)])
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output
    
    def _forward_single(self, datapoint):
        output = self.biases.clone()
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                output[i] += F.conv2d(datapoint[:, j:j+1, :, :], torch.flip(self.kernels[i:i+1, j:j+1, :, :], dims=[2, 3]), padding="valid")[0, 0, :, :]
        return output
    
    def backward(self, dCdy, learning_rate=0.001, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy, learning_rate=learning_rate)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy, learning_rate=learning_rate)
        kernel_gradient = torch.zeros_like(self.kernels, device=self.device, dtype=self.data_type)
        dCdx = torch.zeros_like(self.input, device=self.device, dtype=self.data_type)
        batch_size = self.input.shape[0]
        for batch in range(batch_size):
            dCdx_batch, kernel_gradient_batch = self._backward_single(dCdy[batch:batch+1])
            kernel_gradient += kernel_gradient_batch
            dCdx[batch] = dCdx_batch
        
        self.biases -= learning_rate * dCdy.mean(dim=0)
        self.kernels -= learning_rate / batch_size * kernel_gradient
        return dCdx


    def _backward_single(self, dCdy_batch):
        kernel_gradient_batch = torch.zeros_like(self.kernels, device=self.device, dtype=self.data_type)
        dCdx_batch = torch.zeros(size=self.input_shape, device=self.device, dtype=self.data_type)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                kernel_gradient_batch[i, j] = F.conv2d(self.input[:, j:j+1, :, :], torch.flip(dCdy_batch[:, i:i+1, :, :], dims=[2, 3]), padding="valid")[0, 0, :, :]
                dCdx_batch[j] += F.conv2d(dCdy_batch[:, i:i+1, :, :], self.kernels[i:i+1, j:j+1, :, :], padding=[self.kernel_size - 1, self.kernel_size - 1])[0, 0, :, :]
        return dCdx_batch, kernel_gradient_batch

