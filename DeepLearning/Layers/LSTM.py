import torch
from math import sqrt

from .Base import Base


class LSTM(Base):
    def __init__(self, output_shape, hidden_size, activation=None, normalisation=None, **kwargs):
        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "RNN"
        self.hidden_size = hidden_size

    def initialise_layer(self):
        input_size = self.input_shape[2]
        output_size = self.input_shape[-1]
        self.wf = (torch.rand(input_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.uf = (torch.rand(self.hidden_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.bf = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wi = (torch.rand(input_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.ui = (torch.rand(self.hidden_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.bi = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wc = (torch.rand(input_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.uc = (torch.rand(self.hidden_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.bc = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wo = (torch.rand(input_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.uo = (torch.rand(self.hidden_size, self.hidden_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.bo = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wy = (torch.rand(self.hidden_size, output_size, dtype=self.data_type, device=self.device) * 2 - 1) * sqrt(6 / (input_size + self.hidden_size))
        self.by = torch.zeros((output_size), dtype=self.data_type, device=self.device)
        self.nparams = output_size + self.hidden_size * output_size + 4 * input_size * self.hidden_size + 4 * self.hidden_size + 4 * self.hidden_size ** 2

    """
    input.shape = (batch_size, sequence_length, input_size)
    output.shape = (batch_size, sequence_length, output_size) or (batch_size, output_size)
    """
    def forward(self, input, training=False, **kwargs):
        self.input = input
        self.forget_gates = {}
        self.input_gates = {}
        self.candidate_gates = {}
        self.output_gates = {}
        batch_size, seq_len, _ = input.size()
        self.cell_states = {-1: torch.zeros((batch_size, self.hidden_size), dtype=self.data_type, device=self.device)}
        self.hidden_states = {-1: torch.zeros((batch_size, self.hidden_size), dtype=self.data_type, device=self.device)}
        if len(self.output_shape) == 3: self.output = torch.zeros((batch_size, seq_len, self.output_shape[2]), dtype=self.data_type, device=self.device)
        for t in range(seq_len):
            x_t = input[:, t]
            h_t_prev = self.hidden_states[t - 1]
            self.forget_gates[t] = self._sigmoid(x_t @ self.wf + h_t_prev @ self.uf + self.bf)
            self.input_gates[t] = self._sigmoid(x_t @ self.wi + h_t_prev @ self.ui + self.bi)
            self.candidate_gates[t] = self._tanh(x_t @ self.wc + h_t_prev @ self.uc + self.bc)
            self.output_gates[t] = self._sigmoid(x_t @ self.wo + h_t_prev @ self.uo + self.bo)

            self.cell_states[t] = self.forget_gates[t] * self.cell_states[t - 1].squeeze() + self.input_gates[t] * self.candidate_gates[t]
            self.hidden_states[t] = self.output_gates[t] * self._tanh(self.cell_states[t])

            if len(self.output_shape) == 3: self.output[:, t] = self.hidden_states[t] @ self.wy + self.by
        
        if len(self.output_shape) == 2: self.output = self.hidden_states[-1] @ self.wy + self.by
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output
    
    def _sigmoid(self, input, derivative=False):
        if derivative:
            return input * (1 - input)
        return 1 / (1 + torch.exp(-input))
    
    def _tanh(self, input, derivative=False):
        if derivative:
            return 1 - input ** 2
        return torch.tanh(input)

    def backward(self, dCdy, **kwargs):
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)

        
        return dCdx
    
    def get_parameters(self):
        return 
