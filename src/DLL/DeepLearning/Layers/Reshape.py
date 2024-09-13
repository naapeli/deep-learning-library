from .Base import Base


class Flatten(Base):
    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape, **kwargs)
        self.name = "Reshape"
    
    def forward(self, input, **kwargs):
        self.input = input
        return input.reshape(input.shape[0], *self.output_shape[1:])
    
    def backward(self, dCdy, **kwargs):
        return dCdy.reshape(*self.input.shape)
