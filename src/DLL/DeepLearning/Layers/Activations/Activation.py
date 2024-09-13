from ..Base import Base


class Activation(Base):
    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape, output_shape, **kwargs)
        assert self.activation is None, "Activation layer must not have an activation function"
        assert self.normalisation is None, "Activation layer must not have a normalisation layer"
        self.name = "Activation"
    
    def set_output_shape(self, output_shape):
        self.output_shape = output_shape
        self.input_shape = output_shape

    def summary(self):
        return f"{self.name} - Output: ({self.output_shape})"
