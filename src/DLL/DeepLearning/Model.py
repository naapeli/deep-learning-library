import torch
from math import floor

from .Layers.Input import Input
from .Losses.MSE import mse
from .Optimisers.ADAM import Adam
from ..Data.DataReader import DataReader
from ..Data.Metrics import calculate_metrics, _round_dictionary


class Model:
    def __init__(self, input_shape, data_type=torch.float32, device=torch.device("cpu"), **kwargs):
        self.layers = [Input(input_shape, device=device, data_type=data_type)]
        self.optimiser = None
        self.loss = mse()
        self.data_type = data_type
        self.device = device

    def add(self, layer):
        layer.input_shape = self.layers[-1].output_shape
        layer.data_type = self.data_type
        layer.device = self.device
        layer.initialise_layer()
        self.layers.append(layer)

    def compile(self, optimiser=Adam(), loss=mse(), metrics=["loss"]):
        self.optimiser = optimiser
        parameters = [parameter for layer in self.layers for parameter in layer.get_parameters()]
        self.optimiser.initialise_parameters(parameters)
        self.loss = loss
        self.metrics = metrics
    
    def summary(self):
        print("Model summary:")
        total_params = 0
        for layer in self.layers:
            print(layer.summary())
            total_params += layer.get_nparams()
        print(f"Total number of parameters: {total_params}")
    
    def predict(self, input, training=False):
        current = input
        for layer in self.layers:
            current = layer.forward(current, training=training)
        return current
    
    def backward(self, initial_gradient, training=False):
        reversedLayers = reversed(self.layers)
        gradient = initial_gradient
        for layer in reversedLayers:
            gradient = layer.backward(gradient, training=training)

    """
    X.shape = (data_length, input_shape)
    Y.shape = (data_length, output_shape)
    """
    def fit(self, X, Y, val_data=None, epochs=10, callback_frequency=1, batch_size=64, shuffle_every_epoch=True, shuffle_data=True, verbose=False):
        history = {metric: torch.zeros(floor(epochs / callback_frequency), dtype=self.data_type) for metric in self.metrics}
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)
        for epoch in range(epochs):
            for x, y in data_reader.get_data():
                predictions = self.predict(x, training=True)
                initial_gradient = self.loss.gradient(predictions, y)
                self.backward(initial_gradient, training=True)
                self.optimiser.update_parameters()
            if epoch % callback_frequency == 0:
                values = calculate_metrics(data=(self.predict(X), Y), metrics=self.metrics, loss=self.loss.loss)
                if val_data is not None:
                    val_values = calculate_metrics(data=(self.predict(val_data[0]), val_data[1]), metrics=self.metrics, loss=self.loss.loss)
                    values |= val_values
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        return history
