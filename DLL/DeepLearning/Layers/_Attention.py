import torch

from ._BaseLayer import BaseLayer
from . import Dense
from .Activations import SoftMax
from .Regularisation import Dropout
from .Activations._Activation import Activation
from .Regularisation._BaseRegularisation import BaseRegularisation


class MultiHeadAttention(BaseLayer):
    """
    The multi head attention layer.

    Args:
        output_shape (tuple[int]): The output_shape of the model not containing the batch_size. Must be a tuple of positive integers (seq_len, output_dim) or (seq_len,). The returned tensor is of shape (n_samples, seq_len, output_dim) if len(output_shape) == 2 else (n_samples, seq_len). The input must have the same shape, where (seq_len,) corresponds to inputs of shape (seq_len, 1).
        n_heads (int): The number of heads used in the layer. The output dimension must be divisible by n_heads.
        use_mask (bool): Determines if a mask is used to make the model only consider past tokens. Must be a boolean.
        dropout (float): The probability of an attention weight being dropped out. Must be in range [0, 1). Defaults to 0.0.
        activation (:ref:`activations_section_label` | None, optional): The activation used after this layer. If is set to None, no activation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
        normalisation (:ref:`regularisation_layers_section_label` | None, optional): The regularisation layer used after this layer. If is set to None, no regularisation is used. Defaults to None. If both activation and regularisation is used, the regularisation is performed first in the forward propagation.
    """
    def __init__(self, output_shape, n_heads=1, use_mask=True, dropout=0.0, activation=None, normalisation=None, **kwargs):
        if not isinstance(activation, Activation) and activation is not None:
            raise ValueError("activation must be from DLL.DeepLearning.Layers.Activations or None.")
        if not isinstance(normalisation, BaseRegularisation) and normalisation is not None:
            raise ValueError("normalisation must be from DLL.DeepLearning.Layers.Regularisation or None.")
        if not isinstance(use_mask, bool):
            raise TypeError("use_mask must be a boolean.")
        if not isinstance(dropout, int | float) or dropout < 0 or 1 <= dropout:
            raise ValueError("dropout must be in range [0, 1).")
        if not isinstance(n_heads, int) or n_heads < 1:
            raise ValueError("n_heads must be a positive integer.")

        super().__init__(output_shape, activation=activation, normalisation=normalisation, **kwargs)
        if len(self.output_shape) not in (1, 2):
            raise ValueError("output_shape must be of the form (seq_len, output_dim) or (seq_len,).")
        if self._output_dim() % n_heads != 0:
            raise ValueError("output_dimension must be divisible by n_heads")
        self.name = "MultiHeadAttention"
        self.use_mask = use_mask
        self.mask = None
        self.dropout = dropout
        self.attention_dropout = None
        self.n_heads = n_heads

    def _output_dim(self):
        return self.output_shape[1] if len(self.output_shape) == 2 else 1

    def initialise_layer(self, input_shape, data_type, device):
        """
        :meta private:
        """
        output_dim = self._output_dim()
        if input_shape[1] != output_dim:
            raise ValueError("input_shape must be the same as the output_shape.")
        if input_shape[0] != self.output_shape[0]:
            raise ValueError("Input_shape must have the same seq_len as the output_shape.")

        super().initialise_layer(input_shape, data_type, device)
        self.softmax = SoftMax()
        seq_len = input_shape[0]

        # project the input to the queries, keys and values
        self.input_linear = Dense((seq_len, 3 * output_dim), bias=False)
        self.input_linear.initialise_layer(input_shape, data_type, device)
        # mix the concatenated heads
        self.output_linear = Dense((seq_len, output_dim))
        self.output_linear.initialise_layer((seq_len, output_dim), data_type, device)
        self.nparams = 3 * output_dim * output_dim + output_dim * output_dim + output_dim

        if self.use_mask:
            self.mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len) == 0
        if self.dropout > 0:
            self.attention_dropout = Dropout(self.dropout)
            self.attention_dropout.initialise_layer(input_shape=(self.n_heads, seq_len, seq_len), data_type=data_type, device=device)

    def forward(self, input, training=False, **kwargs):
        """
        Applies the attention mechanism on multiple heads.

        .. math::
        
            \\begin{align*}
                y_{\\text{MultiHead}} &= \\text{Concat}(head_1, \\dots, head_{\\text{n_heads}})W_O + b_O,\\\\
                y_{reg} &= f(y_{\\text{MultiHead}}),\\\\
                y_{activ} &= g(y_{reg}),
            \\end{align*}
        
        where :math:`head_i = \\text{softmax}\\left(\\frac{Q_iK_i^T}{\\sqrt{d_{\\text{head}}}}\\right)V_i`, :math:`d_{\\text{head}} = \\text{output_dim} / \\text{n_heads}`, :math:`f` is the possible regularisation function and :math:`g` is the possible activation function. :math:`Q, K` and :math:`V` are the query, key and value matricies, which are obtained by transforming the input with a linear layer and splitting the result on the feature axis. :math:`Q_i, K_i` and :math:`V_i` are the parts of them belonging to head :math:`i`, and :math:`W_O` and :math:`b_O` are the weights and the bias of the output projection. If use_mask is True, the softmax only considers the current and the past tokens.

        Args:
            input (torch.Tensor of shape (n_samples, seq_len, output_dim)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.
            training (bool, optional): The boolean flag deciding if the model is in training mode. Defaults to False.

        Returns:
            torch.Tensor of shape (n_samples, seq_len) if len(output_shape) == 1 else (n_samples, seq_len, output_dim): The output tensor after the transformations with the spesified shape.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor.")
        if input.shape[1:] != self.input_shape:
            raise ValueError(f"input is not the same shape as the spesified input_shape ({input.shape[1:], self.input_shape}).")
        if not isinstance(training, bool):
            raise TypeError("training must be a boolean.")
        
        self.input = input
        batch_size, seq_len, _ = input.size()
        output_dim = self._output_dim()
        head_dim = output_dim // self.n_heads

        # split the input to the query, key and the value matricies
        input = self.input_linear.forward(input, training=training)
        Q, K, V  = input.chunk(3, dim=2)
        self.Q = Q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        self.K = K.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        self.V = V.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        output = (self.Q @ self.K.transpose(-2, -1)) * head_dim ** (-0.5)
        if self.mask is not None: output = output.masked_fill(self.mask, float('-inf'))
        output = self.softmax.forward(output, training=training)
        if self.attention_dropout is not None: output = self.attention_dropout.forward(output, training=training)
        self.attention_scores = output
        output = output @ self.V
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, output_dim)
        output = self.output_linear.forward(output, training=training)

        if self.normalisation: output = self.normalisation.forward(output, training=training)
        if self.activation: output = self.activation.forward(output, training=training)
        if len(self.output_shape) == 1: output = output.squeeze(dim=2)  # If the output_shape is a 2d tensor, remove the last dimension
        return output

    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of shape (n_samples, seq_len, output_dim) if len(output_shape) == 2 else (n_samples, seq_len)): The gradient given by the next layer.

        Returns:
            torch.Tensor of shape (n_samples, seq_len, output_dim): The new gradient after backpropagation through the layer.
        """
        if not isinstance(dCdy, torch.Tensor):
            raise TypeError("dCdy must be a torch.Tensor.")
        if dCdy.shape[1:] != self.output_shape:
            raise ValueError(f"dCdy is not the same shape as the spesified output_shape ({dCdy.shape[1:], self.output_shape}).")

        if len(self.output_shape) == 1: dCdy = dCdy.unsqueeze(dim=2)  # If the output shape was a 2d tensor, add an extra dimension to the end
        if self.activation: dCdy = self.activation.backward(dCdy)
        if self.normalisation: dCdy = self.normalisation.backward(dCdy)
        dCdy = self.output_linear.backward(dCdy)

        batch_size, seq_len, output_dim = dCdy.size()
        head_dim = output_dim // self.n_heads
        dCdy = dCdy.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        dCdV = self.attention_scores.transpose(-1, -2) @ dCdy
        dCdy = dCdy @ self.V.transpose(-1, -2)
        if self.attention_dropout is not None: dCdy = self.attention_dropout.backward(dCdy)
        dCdy = self.softmax.backward(dCdy)
        if self.mask is not None: dCdy = dCdy * (~self.mask)

        dCdy = dCdy * head_dim ** (-0.5)
        dCdQ = dCdy @ self.K
        dCdK = dCdy.transpose(-2, -1) @ self.Q

        dCdQ = dCdQ.transpose(1, 2).reshape(batch_size, seq_len, output_dim)
        dCdK = dCdK.transpose(1, 2).reshape(batch_size, seq_len, output_dim)
        dCdV = dCdV.transpose(1, 2).reshape(batch_size, seq_len, output_dim)
        dCdx = torch.cat([dCdQ, dCdK, dCdV], dim=2)
        dCdx = self.input_linear.backward(dCdx)

        return dCdx
    
    def get_parameters(self):
        """
        :meta private:
        """
        return (*self.input_linear.get_parameters(), *self.output_linear.get_parameters(), *super().get_parameters())
