from ._BaseLayer import BaseLayer


class Input(BaseLayer):
    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape=output_shape, input_shape=output_shape, **kwargs)
        self.name = "Input"

    def summary(self, offset=""):
        output_shape = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape
        return offset + f"{self.name} - Output: ({output_shape})"
