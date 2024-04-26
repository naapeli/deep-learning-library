import numpy as np
from Layers.Base import Base


class Activation(Base):
    def __init__(self, output_size=None, **kwargs):
        super().__init__(output_size, output_size)
        assert self.activation is None, "Activation layer must not have an activation function"
        assert self.normalisation is None, "Activation layer must not have a normalisation layer"
        self.name = "Activation"
    
    def set_output_size(self, output_size):
        self.output_size = output_size
        self.input_size = output_size

    def summary(self):
        return f"{self.name} - Output: ({self.output_size})"
