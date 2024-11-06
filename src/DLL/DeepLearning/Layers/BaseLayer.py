import torch


class BaseLayer:
    def __init__(self, output_shape, input_shape=None, activation=None, normalisation=None, data_type=torch.float32, device=torch.device("cpu")):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output = None
        self.input = None
        self.nparams = 0
        self.name = "base"
        self.activation = activation
        self.normalisation = normalisation
        self.device = device
        self.data_type = data_type
        if self.activation:
            self.activation.initialise_layer(output_shape, data_type, device)
        if self.normalisation:
            self.normalisation.initialise_layer(output_shape, data_type, device)

    def initialise_layer(self, input_shape, data_type, device):
        self.input_shape = input_shape
        self.data_type = data_type
        self.device = device

    def summary(self):
        return f"{self.name} - (Input, Output): ({self.input_shape}, {self.output_shape}) - Parameters: {self.nparams}" + (" - Normalisation: (" + self.normalisation.summary() + ")" if self.normalisation else "") + (" - Activation: " + self.activation.name if self.activation else "")

    def forward(self, input, **kwargs):
        self.input = input
        return self.input

    def backward(self, dCdy, **kwargs):
        return dCdy
    
    def get_nparams(self):
        return self.nparams + (self.normalisation.nparams if self.normalisation else 0)
    
    def get_parameters(self):
        return []
