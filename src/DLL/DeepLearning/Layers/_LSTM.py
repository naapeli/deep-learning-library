import torch

from ._BaseLayer import BaseLayer
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser


class LSTM(BaseLayer):
    """
    The long short-term memory layer for neural networks.

    Args:
        output_shape (int): The number of output features. Must be a non-negative int. If is zero, the returned tensor is of shape (batch_size,) or (batch_size, sequence_length) and if positive, the returned tensor is of shape (batch_size, output_size) or (batch_size, sequence_length, output_size).
        hidden_size (int): The number of features in the hidden state vector. Must be a positive integer.
        return_last (bool): Determines if only the last element or the whole sequence is returned.
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, output_shape, hidden_size, return_last=True, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
        if not isinstance(output_shape, int) or output_shape < 0:
            raise ValueError("output_shape must be a non-negative integer.")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(return_last, bool):
            raise TypeError("return_last must be a boolean.")
        if not isinstance(initialiser, Initialiser):
            raise ValueError('initialiser must be an instance of DLL.DeepLearning.Initialisers')
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")

        super().__init__((output_shape,), activation=activation, normalisation=normalisation, **kwargs)
        self.name = "LSTM"
        self.hidden_size = hidden_size
        self.return_last = return_last
        self.initialiser = initialiser

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if len(input_shape) == 2: input_shape = (input_shape[1],)  # Include only the number of features
        if not isinstance(input_shape, tuple | list) or len(input_shape) != 1:
            raise ValueError("input_shape must be a tuple of length 1.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        super().initialise_layer(input_shape, data_type, device)

        input_size = self.input_shape[0]
        output_size = self.output_shape[0]

        self.wf = self.initialiser.initialise((input_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.uf = self.initialiser.initialise((self.hidden_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.bf = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wi = self.initialiser.initialise((input_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.ui = self.initialiser.initialise((self.hidden_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.bi = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wc = self.initialiser.initialise((input_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.uc = self.initialiser.initialise((self.hidden_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.bc = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wo = self.initialiser.initialise((input_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.uo = self.initialiser.initialise((self.hidden_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.bo = torch.zeros((self.hidden_size), dtype=self.data_type, device=self.device)
        self.wy = self.initialiser.initialise((self.hidden_size, output_size), data_type=self.data_type, device=self.device)
        self.by = torch.zeros((output_size), dtype=self.data_type, device=self.device)

        self.nparams = output_size + self.hidden_size * output_size + 4 * input_size * self.hidden_size + 4 * self.hidden_size + 4 * self.hidden_size ** 2

    def forward(self, input, training=False, **kwargs):
        """
        Calculates the forward propagation of the model using the equations

        .. math::
        
            \\begin{align*}
                f_t &= \\sigma(W_fx_t + U_fh_{t - 1} + b_f),\\\\
                i_t &= \\sigma(W_ix_t + U_ih_{t - 1} + b_i),\\\\
                o_t &= \\sigma(W_ox_t + U_oh_{t - 1} + b_o),\\\\
                \\widetilde{c}_t &= \\text{tanh}(W_cx_t + U_ch_{t - 1} + b_c),\\\\
                c_t &= f_t\\odot c_{t - 1} + i_t\\odot\\widetilde{c}_t,\\\\
                h_t &= o_t\\odot\\text{tanh}(c_t),\\\\
                y_t &= W_yh_t + b_y,\\\\
                y_{reg} &= f(y) \\text{ or } f(y_\\text{sequence_length}),\\\\
                y_{activ} &= g(y_{reg}),
            \\end{align*}

        where :math:`t\in[1,\dots, \\text{sequence_length}]`, :math:`x` is the input, :math:`h_t` is the hidden state, :math:`W` and :math:`U` are the weight matricies, :math:`b` are the biases, :math:`f` is the possible regularisation function and :math:`g` is the possible activation function. Also :math:`\\odot` represents the hadamard or the element-wise product and :math:`\\sigma` represents the sigmoid function.

        Args:
            input (torch.Tensor of shape (batch_size, sequence_length, input_size)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.
            
        Returns:
            torch.Tensor: The output tensor after the transformations with the spesified shape.
            
            .. list-table:: The return shapes of the method depending on the parameters.
                :widths: 10 25
                :header-rows: 1

                * - Parameter
                  - Return Shape
                * - LSTM.output_shape[0] == 0 and LSTM.return_last
                  - (batch_size,)
                * - LSTM.output_shape[0] > 0 and LSTM.return_last
                  - (batch_size, output_size)
                * - LSTM.output_shape[0] == 0 and not LSTM.return_last
                  - (batch_size, sequence_length)
                * - LSTM.output_shape[0] > 0 and not LSTM.return_last
                  - (batch_size, sequence_length, output_size)
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[2:] != self.input_shape:
            raise ValueError(f"Input shape {input.shape[2:]} does not match the expected shape {self.input_shape}.")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        batch_size, seq_len, _ = input.size()
        self.input = input
        self.forget_gates = [0] * seq_len
        self.input_gates = [0] * seq_len
        self.candidate_gates = [0] * seq_len
        self.output_gates = [0] * seq_len

        self.cell_states = {-1: torch.zeros((batch_size, self.hidden_size), dtype=input.dtype, device=input.device)}
        self.hidden_states = {-1: torch.zeros((batch_size, self.hidden_size), dtype=input.dtype, device=input.device)}
        if not self.return_last: self.output = torch.zeros((batch_size, seq_len, self.output_shape[0]), dtype=input.dtype, device=input.device)

        for t in range(seq_len):
            x_t = input[:, t]
            h_t_prev = self.hidden_states[t - 1]

            self.forget_gates[t] = self._sigmoid(x_t @ self.wf + h_t_prev @ self.uf + self.bf)
            self.input_gates[t] = self._sigmoid(x_t @ self.wi + h_t_prev @ self.ui + self.bi)
            self.candidate_gates[t] = self._tanh(x_t @ self.wc + h_t_prev @ self.uc + self.bc)
            self.output_gates[t] = self._sigmoid(x_t @ self.wo + h_t_prev @ self.uo + self.bo)

            self.cell_states[t] = self.forget_gates[t] * self.cell_states[t - 1] + self.input_gates[t] * self.candidate_gates[t]
            self.hidden_states[t] = self.output_gates[t] * self._tanh(self.cell_states[t])

            if not self.return_last: self.output[:, t] = self.hidden_states[t] @ self.wy + self.by
        
        if self.return_last: self.output = self.hidden_states[seq_len - 1] @ self.wy + self.by
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (batch_size, sequence_length, input_size): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)

        self._reset_gradients()
        dCdh_next = torch.zeros_like(self.hidden_states[0], dtype=dCdy.dtype, device=dCdy.device) if not self.return_last else dCdy @ self.wy.T
        dCdc_next = torch.zeros_like(self.cell_states[0], dtype=dCdy.dtype, device=dCdy.device)
        dCdx = torch.zeros_like(self.input, dtype=dCdy.dtype, device=dCdy.device)
        batch_size, seq_len, _ = self.input.size()

        if self.return_last: self.wy.grad += self.hidden_states[seq_len - 1].T @ dCdy
        if self.return_last: self.by.grad += torch.mean(dCdy, axis=0)

        for t in reversed(range(seq_len)):
            if not self.return_last: self.wy.grad += self.hidden_states[t].T @ dCdy[:, t]
            if not self.return_last: self.by.grad += torch.mean(dCdy[:, t], axis=0)
            dCdh_t = dCdh_next + dCdy[:, t] @ self.wy.T if not self.return_last else dCdh_next # hidden state

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
        """
        :meta private:
        """
        return (self.wy, self.wo, self.wc, self.wi, self.wf,
                self.uo, self.uc, self.ui, self.uf,
                self.by, self.bo, self.bc, self.bi, self.bf)
