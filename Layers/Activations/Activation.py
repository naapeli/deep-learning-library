import numpy as np
from Layers.Base import Base


class Activation(Base):
    def __init__(self, output_size, input_size=None, activation=None):
        assert activation == None, "Activation must be None on activation layers"
        assert input_size == output_size, "Input_size must be the same as output_size on activation layers"
        super().__init__(output_size, output_size)
        self.name = "Activation"
