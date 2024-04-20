import torch


class Base:
    def __init__(self, output_size, input_size=None, activation=None, data_type=torch.float32, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.output = None
        self.input = None
        self.nparams = 0
        self.name = "base"
        self.activation = activation
        self.p = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_type = data_type
        if self.activation: self.activation.__init__(output_size)
    
    def initialise_layer(self):
        pass

    def summary(self):
        return f"{self.name} - (Input, Output): ({self.input_size}, {self.output_size}) - Parameters: {self.nparams}" + (" - Activation: " + self.activation.name if self.activation else "")

    def forward(self, input, **kwargs):
        self.input = input
        return self.input

    def backward(self, dCdy, **kwargs):
        return dCdy
