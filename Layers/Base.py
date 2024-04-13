class Base:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.dCdx = 0

    def forward(self, input):
        pass

    def backward(self):
        pass
