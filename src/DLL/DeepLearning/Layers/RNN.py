import torch
from math import sqrt

from .BaseLayer import BaseLayer


class RNN(BaseLayer):
    def __init__(self, output_shape, hidden_size, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "RNN"
        self.hidden_size = hidden_size

    def initialise_layer(self):
        self.ih = torch.normal(mean=0, std=1 / sqrt(self.input_shape[-1] + self.output_shape[-1]), size=(self.hidden_size, self.input_shape[-1]), dtype=self.data_type, device=self.device)
        self.hh = torch.normal(mean=0, std=1 / sqrt(self.input_shape[-1] + self.output_shape[-1]), size=(self.hidden_size, self.hidden_size), dtype=self.data_type, device=self.device)
        self.ho = torch.normal(mean=0, std=1 / sqrt(self.input_shape[-1] + self.output_shape[-1]), size=(self.output_shape[-1], self.hidden_size), dtype=self.data_type, device=self.device)
        self.bh = torch.zeros(self.hidden_size, dtype=self.data_type, device=self.device)
        self.bo = torch.zeros(self.output_shape[-1], dtype=self.data_type, device=self.device)
        self.nparams = self.hidden_size * self.hidden_size + self.hidden_size * self.output_shape[-1] + self.hidden_size * self.input_shape[-1] + self.hidden_size + self.output_shape[-1]

    """
    input.shape = (batch_size, sequence_length, input_size)
    output.shape = (batch_size, sequence_length, output_size) or (batch_size, output_size)
    """
    def forward(self, input, training=False, **kwargs):
        self.input = input
        batch_size, seq_len, _ = input.size()
        self.hiddens = [torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)]
        if len(self.output_shape) == 3: self.output = torch.zeros(batch_size, seq_len, self.output_shape[2], dtype=input.dtype, device=input.device)

        for t in range(seq_len):
            self.hiddens.append(torch.tanh(self.input[:, t] @ self.ih.T + self.hiddens[-1] @ self.hh.T + self.bh))
            if len(self.output_shape) == 3: self.output[:, t] = self.hiddens[-1] @ self.ho.T + self.bo

        if len(self.output_shape) == 2: self.output = self.hiddens[-1] @ self.ho.T + self.bo
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)

        self.hh.grad = torch.zeros_like(self.hh, dtype=dCdy.dtype, device=dCdy.device)
        self.ih.grad = torch.zeros_like(self.ih, dtype=dCdy.dtype, device=dCdy.device)
        self.ho.grad = torch.zeros_like(self.ho, dtype=dCdy.dtype, device=dCdy.device)
        self.bh.grad = torch.zeros_like(self.bh, dtype=dCdy.dtype, device=dCdy.device)
        self.bo.grad = torch.zeros_like(self.bo, dtype=dCdy.dtype, device=dCdy.device)

        batch_size, seq_len, _ = self.input.size()
        dCdh_next = torch.zeros_like(self.hiddens[0], dtype=dCdy.dtype, device=dCdy.device) if len(self.output_shape) == 3 else dCdy @ self.ho
        dCdx = torch.zeros_like(self.input, dtype=dCdy.dtype, device=dCdy.device)

        if len(self.output_shape) == 2: self.ho.grad += dCdy.T @ self.hiddens[-1] # batch_size, output_size --- batch_size, self.hidden_size
        if len(self.output_shape) == 2: self.bo.grad += torch.sum(dCdy, axis=0)
        for t in reversed(range(seq_len)):
            if len(self.output_shape) == 3: self.ho.grad += dCdy[:, t].T @ self.hiddens[t + 1]
            if len(self.output_shape) == 3: self.bo.grad += torch.sum(dCdy[:, t], axis=0)

            dCdh_t = dCdh_next + dCdy[:, t] @ self.ho if len(self.output_shape) == 3 else dCdh_next # batch_size, self.hidden_size + batch_size, output_size --- self.output_shape, self.hidden_size

            dCdtanh = (1 - self.hiddens[t + 1] ** 2) * dCdh_t # batch_size, self.hidden_size

            self.bh.grad += torch.sum(dCdtanh, axis=0)
            self.ih.grad += dCdtanh.T @ self.input[:, t] # batch_size, self.hidden_size --- batch_size, input_size
            self.hh.grad += dCdtanh.T @ self.hiddens[t] # batch_size, self.hidden_size --- batch_size, self.hidden_size

            dCdh_next = dCdtanh @ self.hh # self.hidden_size, self.hidden_size --- batch_size, self.hidden_size
            dCdx[:, t] = dCdtanh @ self.ih # batch_size, self.hidden_size --- self.hidden_size, self.input_shape
        return dCdx
    
    def get_parameters(self):
        return (self.hh, self.ih, self.ho, self.bh, self.bo)
