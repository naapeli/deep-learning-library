from Layers.Input import Input
from Losses.MSE import mse
import torch


class Model:
    def __init__(self, input_size):
        self.layers = [Input(input_size)]
        self.optimiser = None
        self.loss = mse()
        self.device = self.layers[0].device

    def add(self, layer):
        layer.__init__(layer.output_size, input_size=self.layers[-1].output_size, activation=layer.activation)
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
    
    def predict(self, input):
        current = input
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def backward(self, initial_gradient):
        reversedLayers = reversed(self.layers)
        gradient = initial_gradient
        for layer in reversedLayers:
            gradient = layer.backward(gradient, learning_rate=0.1) # self.optimiser.learning_rate

    """
    X.shape = (data_length, input_size)
    """
    def fit(self, X, Y, val_data=None, epochs=100, loss_step=5, batch_size=None):
        errors = torch.zeros(epochs, dtype=torch.float32, device=self.device, requires_grad=False)
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in zip(X, Y):
                predictions = self.predict(x)
                error += self.loss.loss(predictions, y)
                initial_gradient = self.loss.gradient(predictions, y)
                # self.optimiser.gradient(initial_gradient)
                self.backward(initial_gradient)
            error /= len(X)
            errors[epoch] = error
            if epoch % loss_step == 0: print(f"Epoch: {epoch} - Error: {error}")
        return errors
