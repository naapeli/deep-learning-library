from .GroupNormalisation import GroupNorm


class LayerNorm(GroupNorm):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
        self.name = "Layer normalisation"

    def forward(self, input, **kwargs):
        """
        Normalises the input to have zero mean and one variance accross the channel dimension with the following equation:
        
        .. math::
            y = \\gamma\\frac{x - \\mathcal{E}[x]}{\\sqrt{\\text{var}(x) + \\epsilon}} + \\beta,
        
        where :math:`x` is the input, :math:`\\mathcal{E}[x]` is the expected value or the mean accross the channel dimension, :math:`\\text{var}(x)` is the variance accross the variance accross the channel dimension, :math:`\\epsilon` is a small constant and :math:`\\gamma` and :math:`\\beta` are trainable parameters.

        Args:
            input (torch.Tensor of shape (batch_size, channels, ...)): The input to the layer. Must be a torch.Tensor of the spesified shape given by layer.input_shape.

        Returns:
            torch.Tensor: The output tensor after the normalisation with the same shape as the input.
        """

        return super().forward(input, **kwargs)
    
    def backward(self, dCdy, **kwargs):
        """
        Calculates the gradient of the loss function with respect to the input of the layer. Also calculates the gradients of the loss function with respect to the model parameters.

        Args:
            dCdy (torch.Tensor of the same shape as returned from the forward method): The gradient given by the next layer.
            
        Returns:
            torch.Tensor of shape (batch_size, channels, ...): The new gradient after backpropagation through the layer.
        """
        return super().backward(dCdy, **kwargs)
