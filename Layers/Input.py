from Layers.Base import Base


class Input(Base):
    def __init__(self, output_size, **kwargs):
        super().__init__(output_size, output_size)
        self.name = "Input"

    def summary(self):
        return f"{self.name} - Output: ({self.output_size})"
