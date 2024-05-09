import torch
from math import sqrt
from Layers.Base import Base
from Layers.Activations.Tanh import Tanh


class RNN(Base):
    def __init__(self, output_shape, hidden_size, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "RNN"
        self.hidden_size = hidden_size

    def initialise_layer(self):
        self.ih = torch.normal(mean=0, std=1, size=(self.hidden_size, self.input_shape), dtype=self.data_type, device=self.device)
        self.hh = torch.normal(mean=0, std=1, size=(self.hidden_size, self.hidden_size), dtype=self.data_type, device=self.device)
        self.ho = torch.normal(mean=0, std=1, size=(self.output_shape, self.hidden_size), dtype=self.data_type, device=self.device)
        self.bh = torch.zeros(self.hidden_size, dtype=self.data_type, device=self.device)
        self.bo = torch.zeros(self.output_shape, dtype=self.data_type, device=self.device)
        self.nparams = self.hidden_size * self.hidden_size + self.hidden_size * self.output_shape + self.hidden_size * self.input_shape + self.hidden_size + self.output_shape

    """
    input.shape = (batch_size, sequence_length, input_size)
    output.shape = (batch_size, sequence_length, output_size)
    """
    def forward(self, input, training=False, **kwargs):
        self.input = input
        batch_size, seq_len, _ = input.size()
        self.hiddens = [torch.zeros(batch_size, self.hidden_size)]
        self.output = torch.zeros(batch_size, seq_len, self.output_shape)
        for t in range(seq_len):
            self.hiddens.append(torch.tanh(self.input[:, t] @ self.ih.T + self.hiddens[-1] @ self.hh.T + self.bh))
            self.output[:, t] = self.hiddens[-1] @ self.ho.T + self.bo
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        
        self.hh.grad = torch.zeros_like(self.hh)
        self.ih.grad = torch.zeros_like(self.ih)
        self.ho.grad = torch.zeros_like(self.ho)
        self.bh.grad = torch.zeros_like(self.bh)
        self.bo.grad = torch.zeros_like(self.bo)

        batch_size, seq_len, _ = self.input.size()
        dCdh_next = torch.zeros_like(self.hiddens[0])
        dCdx = torch.zeros_like(self.input)
        for t in reversed(range(seq_len)):
            self.ho.grad += dCdy[:, t].T @ self.hiddens[t + 1]
            self.bo.grad += torch.sum(dCdy[:, t], axis=0)

            dCdh_t = dCdh_next + dCdy[:, t] @ self.ho # batch_size, self.hidden_size + batch_size, output_size --- self.output_shape, self.hidden_size
            dCdtanh = (1 - self.hiddens[t + 1] ** 2) * dCdh_t # batch_size, self.hidden_size

            self.bh.grad += torch.sum(dCdtanh, axis=0)
            self.ih.grad += dCdtanh.T @ self.input[:, t] # batch_size, self.hidden_size --- batch_size, input_size
            self.hh.grad += dCdtanh.T @ self.hiddens[t] # batch_size, self.hidden_size --- batch_size, self.hidden_size

            dCdh_next = dCdtanh @ self.hh # self.hidden_size, self.hidden_size --- batch_size, self.hidden_size
            dCdx[:, t] = dCdtanh @ self.ih # batch_size, self.hidden_size --- self.hidden_size, self.input_shape
        return dCdx
    
    def get_parameters(self):
        return (self.hh, self.ih, self.ho, self.bh, self.bo)
