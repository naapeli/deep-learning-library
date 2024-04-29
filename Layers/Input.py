from Layers.Base import Base


class Input(Base):
    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape, output_shape)
        self.name = "Input"

    def summary(self):
        return f"{self.name} - Output: ({self.output_shape})"
