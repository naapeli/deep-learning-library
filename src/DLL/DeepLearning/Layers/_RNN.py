import torch

from ._BaseLayer import BaseLayer
from ..Initialisers import Xavier_Uniform
from ..Initialisers._Initialiser import Initialiser
from ...Exceptions import NotCompiledError
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation


class RNN(BaseLayer):
    """
    The recurrent neural network layer.

    Args:
        output_shape (tuple[int]): The ouput shape by the forward method. Must be tuple containing non-negative ints. Based on the length of the tuple and the return_last parameter, the returned tensor is of shape (n_samples,), (n_samples, sequence_length), (n_samples, n_features) or (n_samples, sequence_length, n_features).
        hidden_size (int): The number of features in the hidden state vector. Must be a positive integer.
        return_last (bool): Determines if only the last element or the whole sequence is returned.
        initialiser (:ref:`initialisers_section_label`, optional): The initialisation method for models weights. Defaults to Xavier_uniform.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, output_shape, hidden_size, return_last=True, initialiser=Xavier_Uniform(), activation=None, normalisation=None, **kwargs):
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
        if return_last and len(output_shape) == 2:
            raise ValueError("return_last should not be True when the output_shape is (seq_len, n_features)")

        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        self.name = "RNN"
        self.hidden_size = hidden_size
        self.return_last = return_last
        self.initialiser = initialiser

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        if not isinstance(input_shape, tuple | list) or len(input_shape) != 2:
            raise ValueError("input_shape must be a tuple of length 2.")
        if not isinstance(data_type, torch.dtype):
            raise TypeError("data_type must be an instance of torch.dtype")
        if not isinstance(device, torch.device):
            raise TypeError('device must be one of torch.device("cpu") or torch.device("cuda")')

        super().initialise_layer(input_shape, data_type, device)

        output_dim = 1 if len(self.output_shape) == 0 or (len(self.output_shape) == 1 and not self.return_last) else self.output_shape[-1]
        self.ih = self.initialiser.initialise((self.hidden_size, self.input_shape[1]), data_type=self.data_type, device=self.device)
        self.hh = self.initialiser.initialise((self.hidden_size, self.hidden_size), data_type=self.data_type, device=self.device)
        self.ho = self.initialiser.initialise((output_dim, self.hidden_size), data_type=self.data_type, device=self.device)
        self.bh = torch.zeros(self.hidden_size, dtype=self.data_type, device=self.device)
        self.bo = torch.zeros(output_dim, dtype=self.data_type, device=self.device)
        self.nparams = self.hidden_size * self.hidden_size + self.hidden_size * output_dim + self.hidden_size * self.input_shape[1] + self.hidden_size + output_dim

    """
    input.shape = (batch_size, sequence_length, input_size)
    output.shape = (batch_size, sequence_length, output_size) or (batch_size, output_size) or (batch_size, sequence_length) or (batch_size,)
    """
    def forward(self, input, training=False, **kwargs):
        """
        Calculates the forward propagation of the model using the equation

        .. math::
        
            \\begin{align*}
                h_t &= \\text{tanh}(x_tW_{ih}^T + h_{t - 1}W_{hh}^T + b_h),\\\\
                y_{t} &= h_tW_o^T + b_o,\\\\
                y_{reg} &= f(y) \\text{ or } f(y_\\text{sequence_length}),\\\\
                y_{activ} &= g(y_{reg}),
            \\end{align*}

        where :math:`t\in[1,\dots, \\text{sequence_length}]`, :math:`x` is the input, :math:`h_t` is the hidden state, :math:`W_{ih}` is the input to hidden weights, :math:`W_{hh}` is the hidden to hidden weights, :math:`b_h` is the hidden bias, :math:`W_o` is the output weights, :math:`b_o` is the output bias, :math:`f` is the possible regularisation function and :math:`g` is the possible activation function.

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
                * - len(RNN.output_shape) == 0 and RNN.return_last
                  - (n_samples,)
                * - len(RNN.output_shape) == 1 and RNN.return_last
                  - (n_samples, RNN.output_shape[1])
                * - len(RNN.output_shape) == 1 and not RNN.return_last
                  - (n_samples, sequence_length)
                * - len(RNN.output_shape) == 2 and not RNN.return_last
                  - (n_samples, sequence_length, RNN.output_shape[1])
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"Input shape {input.shape[1:]} does not match the expected shape {self.input_shape}.")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")

        self.input = input
        batch_size, seq_len, _ = input.size()
        self.hiddens = [torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)]
        output_dim = 1 if len(self.output_shape) == 0 or (len(self.output_shape) == 1 and not self.return_last) else self.output_shape[-1]
        if not self.return_last: self.output = torch.zeros(batch_size, seq_len, output_dim, dtype=input.dtype, device=input.device)

        for t in range(seq_len):
            self.hiddens.append(torch.tanh(self.input[:, t] @ self.ih.T + self.hiddens[-1] @ self.hh.T + self.bh))
            if not self.return_last: self.output[:, t] = self.hiddens[-1] @ self.ho.T + self.bo

        if self.return_last: self.output = self.hiddens[-1] @ self.ho.T + self.bo
        if self.normalisation: self.output = self.normalisation.forward(self.output, training=training)
        if self.activation: self.output = self.activation.forward(self.output)
        if len(self.output_shape) == 0 or (len(self.output_shape) == 1 and not self.return_last): output = output.squeeze(dim=-1)
        return self.output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (n_samples, sequence_length, input_size): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output.shape[1:]:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output.shape[1:]}).")
        
        if len(self.output_shape) == 0 or (len(self.output_shape) == 1 and not self.return_last): dCdy = dCdy.unsqueeze(dim=-1)
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)

        self.hh.grad = torch.zeros_like(self.hh, dtype=dCdy.dtype, device=dCdy.device)
        self.ih.grad = torch.zeros_like(self.ih, dtype=dCdy.dtype, device=dCdy.device)
        self.ho.grad = torch.zeros_like(self.ho, dtype=dCdy.dtype, device=dCdy.device)
        self.bh.grad = torch.zeros_like(self.bh, dtype=dCdy.dtype, device=dCdy.device)
        self.bo.grad = torch.zeros_like(self.bo, dtype=dCdy.dtype, device=dCdy.device)

        _, seq_len, _ = self.input.size()
        dCdh_next = torch.zeros_like(self.hiddens[0], dtype=dCdy.dtype, device=dCdy.device) if not self.return_last else dCdy @ self.ho
        dCdx = torch.zeros_like(self.input, dtype=dCdy.dtype, device=dCdy.device)

        if self.return_last: self.ho.grad += dCdy.T @ self.hiddens[-1] # batch_size, output_size --- batch_size, self.hidden_size
        if self.return_last: self.bo.grad += torch.sum(dCdy, axis=0)
        for t in reversed(range(seq_len)):
            if not self.return_last: self.ho.grad += dCdy[:, t].T @ self.hiddens[t + 1]
            if not self.return_last: self.bo.grad += torch.sum(dCdy[:, t], axis=0)

            dCdh_t = dCdh_next + dCdy[:, t] @ self.ho if not self.return_last else dCdh_next # batch_size, self.hidden_size + batch_size, output_size --- self.output_shape, self.hidden_size

            dCdtanh = (1 - self.hiddens[t + 1] ** 2) * dCdh_t # batch_size, self.hidden_size

            self.bh.grad += torch.sum(dCdtanh, axis=0)
            self.ih.grad += dCdtanh.T @ self.input[:, t] # batch_size, self.hidden_size --- batch_size, input_size
            self.hh.grad += dCdtanh.T @ self.hiddens[t] # batch_size, self.hidden_size --- batch_size, self.hidden_size

            dCdh_next = dCdtanh @ self.hh # self.hidden_size, self.hidden_size --- batch_size, self.hidden_size
            dCdx[:, t] = dCdtanh @ self.ih # batch_size, self.hidden_size --- self.hidden_size, self.input_shape
        return dCdx

    def get_parameters(self):
        """
        :meta private:
        """
        return (self.hh, self.ih, self.ho, self.bh, self.bo)
    
    # def summary(self, offset=""):
    #     if not hasattr(self, "input_shape"):
    #         raise NotCompiledError("layer must be initialized correctly before calling layer.summary().")

    #     input_shape = "(seq_len, " + str(self.input_shape[0]) + ")"
    #     output_shape = str(self.output_shape[0]) if self.return_last else "(seq_len, " + str(self.output_shape[0]) + ")"
    #     params_summary = " - Parameters: " + str(self.nparams) if self.nparams > 0 else ""
    #     sublayer_offset = offset + "    "
    #     normalisation_summary = ("\n" + self.normalisation.summary(sublayer_offset)) if self.normalisation else ""
    #     activation_summary = ("\n" + self.activation.summary(sublayer_offset)) if self.activation else ""
    #     return offset + f"{self.name} - (Input, Output): ({input_shape}, {output_shape})" + params_summary + normalisation_summary + activation_summary
