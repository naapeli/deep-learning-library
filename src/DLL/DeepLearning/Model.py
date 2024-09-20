import torch
from math import floor

from .Layers.Input import Input
from .Losses.MSE import mse
from .Optimisers.ADAM import Adam
from ..Data.DataReader import DataReader
from ..Data.Metrics import accuracy


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
    def fit(self, X, Y, val_data=None, epochs=10, callback_frequency=1, batch_size=64, shuffle_every_epoch=True, shuffle_data=True):
        history = {metric: torch.zeros(floor(epochs / callback_frequency), dtype=self.data_type) for metric in self.metrics}
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)
        for epoch in range(epochs):
            for x, y in data_reader.get_data():
                predictions = self.predict(x, training=True)
                initial_gradient = self.loss.gradient(predictions, y)
                self.backward(initial_gradient, training=True)
                self.optimiser.update_parameters()
            if epoch % callback_frequency == 0:
                values = self._calculate_metrics(data=(X, Y), val_data=val_data)
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                print(f"Epoch: {epoch + 1} - Metrics: {self._round_dictionary(values)}")
        return history
    
    def _round_dictionary(self, values):
        return {key: "{:0.4f}".format(value) for key, value in values.items()}
    
    def _calculate_metrics(self, data=None, val_data=None):
        values = {}
        if data:
            X, Y = data
            predictions = self.predict(X)
        if val_data:
            x_val, y_val = val_data
            val_predictions = self.predict(x_val)
        for metric in self.metrics:
            if val_data and metric[:3] == "val":
                if metric == "val_loss":
                    metric_value = self.loss.loss(val_predictions, y_val).item()
                elif metric == "val_accuracy":
                    metric_value = accuracy(val_predictions, y_val)
            elif metric == "loss":
                metric_value = self.loss.loss(predictions, Y).item()
            elif metric == "accuracy":
                metric_value = accuracy(predictions, Y)
            else:
                print(f"Metric {metric} not implemented")

            values[metric] = metric_value
        return values
