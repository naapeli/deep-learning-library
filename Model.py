from Layers.Dense import Dense
from Layers.Activations.Tanh import Tanh
from Losses.MSE import mse
import numpy as np

class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, input):
        current = input
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def backward(self, initial_gradient):
        reversedLayers = reversed(self.layers)
        gradient = initial_gradient
        for layer in reversedLayers:
            gradient = layer.backward(gradient, learning_rate=0.1)


    def fit(self, X, Y, epochs, loss, optimiser):
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in zip(X, Y):
                prediction = self.predict(x)
                error += loss.loss(prediction, y)
                initial_gradient = loss.gradient(prediction, y)#optimiser.gradient(loss, prediction, y)
                self.backward(initial_gradient)
            if epoch % 500 == 0: print(f"Epoch: {epoch} - Error: {error / len(X)}")
            



model = Model([Dense(2, 3), Tanh(3), Dense(3, 1), Tanh(1)])
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
y = np.array([[0], [1], [1], [0]]).reshape(4, 1, 1)
model.fit(x, y, 10000, mse, None)
print(model.predict(np.array([0, 0]).reshape(1, 2, 1)))
print(model.predict(np.array([0, 1]).reshape(1, 2, 1)))
print(model.predict(np.array([1, 0]).reshape(1, 2, 1)))
print(model.predict(np.array([1, 1]).reshape(1, 2, 1)))
print(model.predict(np.array([0.5, 0.5]).reshape(1, 2, 1)))
