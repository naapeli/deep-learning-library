from ..BaseLayer import BaseLayer


class Activation(BaseLayer):
    def __init__(self, output_shape=None, **kwargs):
        output_shape = (output_shape,) if output_shape is not None else output_shape
        super().__init__(output_shape, output_shape, **kwargs)
        assert self.activation is None, "Activation layer must not have an activation function"
        assert self.normalisation is None, "Activation layer must not have a normalisation layer"
        self.name = "Activation"

    def initialise_layer(self, input_shape, data_type, device):
        self.output_shape = input_shape
        super().initialise_layer(input_shape, data_type, device)

    def summary(self):
        return f"{self.name} - Output: ({self.output_shape})"
