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
    
    def backward(self, initial_gradient, training=False):
        reversedLayers = reversed(self.layers)
        gradient = initial_gradient
        for layer in reversedLayers:
            gradient = layer.backward(gradient, learning_rate=0.1, training=training) # self.optimiser.learning_rate

    """
    X.shape = (data_length, input_size)
    Y.shape = (data_length, output_size)
    """
    def fit(self, X, Y, val_data=None, metrics=["loss"], epochs=100, loss_step=5, batch_size=64, new_shuffle_per_epoch=False, shuffle_data=True):
        errors = torch.zeros(floor(epochs / loss_step), dtype=self.data_type, device=self.device, requires_grad=False)
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, new_shuffle_per_epoch=new_shuffle_per_epoch)
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in data_reader.get_data():
                predictions = self.predict(x, training=True)
                error += self.loss.loss(predictions, y)
                initial_gradient = self.loss.gradient(predictions, y)
                # self.optimiser.gradient(initial_gradient)
                self.backward(initial_gradient, training=True)
            error /= len(X)
            if epoch % loss_step == 0:
                errors[int(epoch / loss_step)] = error
                print(f"Epoch: {epoch} - Error: {error}")
        return errors
