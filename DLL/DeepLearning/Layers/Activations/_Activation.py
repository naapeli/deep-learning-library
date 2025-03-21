from .._BaseLayer import BaseLayer


class Activation(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(None, None, **kwargs)

        if self.output_shape is not None:
            raise ValueError("The output_shape should be None for activation layers.")
        if self.activation is not None:
            raise ValueError("Activation layer must not have an activation function.")
        if self.normalisation is not None:
            raise ValueError("Activation layer must not have a normalisation layer.")

        self.name = "Activation"

    def initialise_layer(self, input_shape, data_type, device):
        self.output_shape = input_shape
        super().initialise_layer(input_shape, data_type, device)

    def summary(self, offset=""):
        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        return offset + f"{self.name} - Output: ({output_shape})"
