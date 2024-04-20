from Layers.Input import Input
from Layers.Activations import Activation
from Losses.MSE import mse
from Data.DataReader import DataReader
import torch
from math import floor


class Model:
    def __init__(self, input_size, data_type=torch.float32, **kwargs):
        self.layers = [Input(input_size)]
        self.optimiser = None
        self.loss = mse()
        self.data_type = data_type
        self.device = self.layers[0].device

    def add(self, layer):
        layer.input_size = self.layers[-1].output_size
        layer.initialise_layer()
        layer.data_type = self.data_type
        self.layers.append(layer)

    def compile(self, optimiser=None, loss=mse()):
        self.optimiser = optimiser
        self.loss = loss
    
    def summary(self):
        print("Model summary:")
        total_params = 0
        for layer in self.layers:
            print(layer.summary())
            total_params += layer.nparams
        print(f"Total number of parameters: {total_params}")
    
    def predict(self, input, training=False):
        current = input
        for layer in self.layers:
            current = layer.forward(current, training=training)
        return current
    
    def backward(self, initial_gradient, learning_rate=0.001, training=False):
        reversedLayers = reversed(self.layers)
        gradient = initial_gradient
        for layer in reversedLayers:
            gradient = layer.backward(gradient, learning_rate=learning_rate, training=training) # self.optimiser.learning_rate

    """
    X.shape = (data_length, input_size)
    Y.shape = (data_length, output_size)
    """
    def fit(self, X, Y, val_data=None, metrics=["loss"], epochs=10, loss_step=1, batch_size=64, learning_rate=0.001, new_shuffle_per_epoch=False, shuffle_data=True):
        history = {metric: torch.zeros(floor(epochs / loss_step), dtype=self.data_type, device=self.device, requires_grad=False) for metric in metrics}
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, new_shuffle_per_epoch=new_shuffle_per_epoch)
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in data_reader.get_data():
                predictions = self.predict(x, training=True)
                error += self.loss.loss(predictions, y)
                initial_gradient = self.loss.gradient(predictions, y)
                # self.optimiser.gradient(initial_gradient)
                self.backward(initial_gradient, learning_rate=learning_rate, training=True)
            error /= len(X)
            if epoch % loss_step == 0:
                values = self._calculate_metrics(y, predictions, metrics, val_data=val_data)
                for metric, value in values.items():
                    history[metric][int(epoch / loss_step)] = value
                print(f"Epoch: {epoch} - Metrics: {values}")
        return history
    
    def _calculate_metrics(self, Y, predictions, metrics, val_data=None):
        values = {}
        for metric in metrics:
            if val_data and metric[:3] == "val":
                x_val, y_val = val_data
                if metric == "val_loss":
                    val_predictions = self.predict(x_val)
                    metric_value = self.loss.loss(val_predictions, y_val).item()

            elif metric == "loss":
                metric_value = self.loss.loss(predictions, Y).item()

            values[metric] = metric_value
        return values
