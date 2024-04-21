import torch


class Base:
    def __init__(self, output_size, input_size=None, activation=None, normalisation=None, data_type=torch.float32, device=torch.device("cpu"), **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.output = None
        self.input = None
        self.nparams = 0
        self.name = "base"
        self.activation = activation
        self.normalisation = normalisation
        self.p = None
        self.device = device
        self.data_type = data_type
        if self.activation: self.activation.__init__(output_size)
        if self.normalisation: self.normalisation.__init__(output_size, patience=self.normalisation.patience)
    
    def initialise_layer(self):
        pass

    def summary(self):
        return f"{self.name} - (Input, Output): ({self.input_size}, {self.output_size}) - Parameters: {self.nparams}" + (" - Normalisation: (" + self.normalisation.summary() + ")" if self.normalisation else "") + (" - Activation: " + self.activation.name if self.activation else "")

    def forward(self, input, **kwargs):
        self.input = input
        return self.input

    def backward(self, dCdy, **kwargs):
        return dCdy
    
    def get_nparams(self):
        return self.nparams + (self.normalisation.nparams if self.normalisation else 0)
