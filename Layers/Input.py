from Layers.Base import Base


class Input(Base):
    def __init__(self, output_size, input_size=None, activation=None):
        assert activation == None, "Activation must be None on an Input layer"
        assert input_size == None, "Input_size must be None on an Input layer"
        super().__init__(output_size, output_size)
        self.name = "Input"
