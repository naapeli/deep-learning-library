from ._BaseLayer import BaseLayer


class Input(BaseLayer):
    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape=output_shape, input_shape=output_shape, **kwargs)
        self.name = "Input"

    def summary(self):
        return f"{self.name} - Output: ({self.output_shape})"
