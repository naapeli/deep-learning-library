import numpy as np
from Layers.Base import Base


class Activation(Base):
    def __init__(self, output_size=1, **kwargs):
        super().__init__(output_size, output_size)
        assert self.activation is None, "Activation layer must not have an activation function"
        self.name = "Activation"

    def summary(self):
        return f"{self.name} - Output: ({self.output_size})"
