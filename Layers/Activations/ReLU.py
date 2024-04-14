import numpy as np
from Layers.Base import Base


class ReLU(Base):
    def __init__(self, output_size, input_size=1):
        super().__init__(output_size, output_size)
        self.name = "ReLU"

    def forward(self, input):
        self.input = input
        self.output = np.maximum(self.input, 0)
        return self.output
    
    def backward(self, dCdy, learning_rate=0.001):
        dCdx = dCdy * (self.input > 0)
        return dCdx
