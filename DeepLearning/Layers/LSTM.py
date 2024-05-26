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
        self.wc.requires_grad = True
        self.wc.retain_grad()
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

            self.cell_states[t] = self.forget_gates[t] * self.cell_states[t - 1] + self.input_gates[t] * self.candidate_gates[t]
            self.hidden_states[t] = self.output_gates[t] * self._tanh(self.cell_states[t])

            if len(self.output_shape) == 3: self.output[:, t] = self.hidden_states[t] @ self.wy + self.by
        
        if len(self.output_shape) == 2: self.output = self.hidden_states[seq_len - 1] @ self.wy + self.by
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, **kwargs):
        # RESULT FOR dCdx DIFFERS < 5 * 10^-4 FROM THE PYTORCH AUTOGRAD RESULT AND IS THUS NOT EXACT!!!
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)

        self._reset_gradients()
        dCdh_next = torch.zeros_like(self.hidden_states[0], dtype=self.data_type, device=self.device) if len(self.output_shape) == 3 else dCdy @ self.wy.T
        dCdc_next = torch.zeros_like(self.cell_states[0], dtype=self.data_type, device=self.device)
        dCdx = torch.zeros_like(self.input, dtype=self.data_type, device=self.device)
        batch_size, seq_len, _ = self.input.size()

        if len(self.output_shape) == 2: self.wy.grad += self.hidden_states[seq_len - 1].T @ dCdy
        if len(self.output_shape) == 2: self.by.grad += torch.mean(dCdy, axis=0)

        for t in reversed(range(seq_len)):
            if len(self.output_shape) == 3: self.wy.grad += self.hidden_states[t].T @ dCdy[:, t]
            if len(self.output_shape) == 3: self.by.grad += torch.mean(dCdy[:, t], axis=0)
            dCdh_t = dCdh_next + dCdy[:, t] @ self.wy.T if len(self.output_shape) == 3 else dCdh_next # hidden state

            dCdo = self._tanh(self.cell_states[t]) * dCdh_t # output
            # dCdc_t = self._tanh(self.cell_states[t], derivative = True) * self.output_gates[t] * dCdh_t + dCdc_next # cell state
            dCdc_t = self._tanh(self._tanh(self.cell_states[t]), derivative = True) * self.output_gates[t] * dCdh_t + dCdc_next # cell state

            dCdc_next = dCdc_t * self.forget_gates[t] # next cell state
            dCdf = dCdc_t * self.cell_states[t - 1] # forget
            dCdi = dCdc_t * self.candidate_gates[t] # input
            dCdc = dCdc_t * self.input_gates[t] # candidate
            
            # activation derivatives
            dCdsig_o = dCdo * self._sigmoid(self.output_gates[t], derivative=True)
            dCdsig_c = dCdc * self._tanh(self.candidate_gates[t], derivative=True)
            dCdsig_i = dCdi * self._sigmoid(self.input_gates[t], derivative=True)
            dCdsig_f = dCdf * self._sigmoid(self.forget_gates[t], derivative=True)
            
            # next hidden state derivative
            dCdh_next = dCdsig_o @ self.uo.T # batch_size, hidden_size --- (hidden_size, hidden_size).T
            dCdh_next += dCdsig_c @ self.uc.T
            dCdh_next += dCdsig_i @ self.ui.T
            dCdh_next += dCdsig_f @ self.uf.T

            # output derivatives
            dCdx[:, t] += dCdsig_o @ self.wo.T # batch_size, hidden_size --- (input_size, hidden_size).T
            dCdx[:, t] += dCdsig_c @ self.wc.T
            dCdx[:, t] += dCdsig_i @ self.wi.T
            dCdx[:, t] += dCdsig_f @ self.wf.T

            # parameter updates
            self.wo.grad += self.input[:, t].T @ dCdsig_o # (batch_size, input_size).T --- batch_size, hidden_size
            self.wc.grad += self.input[:, t].T @ dCdsig_c
            self.wi.grad += self.input[:, t].T @ dCdsig_i
            self.wf.grad += self.input[:, t].T @ dCdsig_f

            self.uo.grad += self.hidden_states[t - 1].T @ dCdsig_o # (batch_size, hidden_size) --- batch_size, hidden_size
            self.uc.grad += self.hidden_states[t - 1].T @ dCdsig_c
            self.ui.grad += self.hidden_states[t - 1].T @ dCdsig_i
            self.uf.grad += self.hidden_states[t - 1].T @ dCdsig_f

            self.bo.grad += torch.mean(dCdsig_o, dim=0)
            self.bc.grad += torch.mean(dCdsig_c, dim=0)
            self.bi.grad += torch.mean(dCdsig_i, dim=0)
            self.bf.grad += torch.mean(dCdsig_f, dim=0)
        return dCdx
    
    def _sigmoid(self, input, derivative=False):
        if derivative:
            return input * (1 - input)
        return 1 / (1 + torch.exp(-input))
    
    def _tanh(self, input, derivative=False):
        if derivative:
            return 1 - input ** 2
        return torch.tanh(input)
    
    def _reset_gradients(self):
        self.wf.grad = torch.zeros_like(self.wf, dtype=self.data_type, device=self.device)
        self.uf.grad = torch.zeros_like(self.uf, dtype=self.data_type, device=self.device)
        self.bf.grad = torch.zeros_like(self.bf, dtype=self.data_type, device=self.device)
        self.wi.grad = torch.zeros_like(self.wi, dtype=self.data_type, device=self.device)
        self.ui.grad = torch.zeros_like(self.ui, dtype=self.data_type, device=self.device)
        self.bi.grad = torch.zeros_like(self.bi, dtype=self.data_type, device=self.device)
        self.wc.grad = torch.zeros_like(self.wc, dtype=self.data_type, device=self.device)
        self.uc.grad = torch.zeros_like(self.uc, dtype=self.data_type, device=self.device)
        self.bc.grad = torch.zeros_like(self.bc, dtype=self.data_type, device=self.device)
        self.wo.grad = torch.zeros_like(self.wo, dtype=self.data_type, device=self.device)
        self.uo.grad = torch.zeros_like(self.uo, dtype=self.data_type, device=self.device)
        self.bo.grad = torch.zeros_like(self.bo, dtype=self.data_type, device=self.device)
        self.wy.grad = torch.zeros_like(self.wy, dtype=self.data_type, device=self.device)
        self.by.grad = torch.zeros_like(self.by, dtype=self.data_type, device=self.device)
    
    def get_parameters(self):
        return (self.wy, self.wo, self.wc, self.wi, self.wf,
                self.uo, self.uc, self.ui, self.uf,
                self.by, self.bo, self.bc, self.bi, self.bf)
