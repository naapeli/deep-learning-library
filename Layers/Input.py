import numpy as np
from Layers.Base import Base


class Input(Base):
    def __init__(self, output_size, input_size=1):
        super().__init__(output_size, output_size)
        self.name = "Input"
