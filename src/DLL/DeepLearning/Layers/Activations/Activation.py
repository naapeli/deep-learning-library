from ..BaseLayer import BaseLayer


class Activation(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(None, None, **kwargs)
        assert self.output_shape == None, "The output_shape should be None for activation layers."
        assert self.activation is None, "Activation layer must not have an activation function"
        assert self.normalisation is None, "Activation layer must not have a normalisation layer"
        self.name = "Activation"

    def initialise_layer(self, input_shape, data_type, device):
        self.output_shape = input_shape
        super().initialise_layer(input_shape, data_type, device)

    def summary(self):
        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        return f"{self.name} - Output: ({output_shape})"
