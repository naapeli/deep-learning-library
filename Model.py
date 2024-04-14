from Layers.Dense import Dense
from Layers.Activations.Tanh import Tanh
from Layers.Input import Input
from Losses.MSE import mse
import numpy as np

class Model:
    def __init__(self, input_size):
        self.layers = [Input(input_size)]
        self.optimiser = None
        self.loss = mse()

    def add(self, layer):
        layer.__init__(layer.output_size, input_size=self.layers[-1].output_size)
        self.layers.append(layer)

    def compile(self, optimiser=None, loss=mse()):
        self.optimiser = optimiser
        self.loss = loss
    
    def summary(self):
        print("Model summary:")
        for layer in self.layers:
            print(layer.summary())
    
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

    def fit(self, X, Y, epochs=100, loss_step=5):
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in zip(X, Y):
                prediction = self.predict(x)
                error += self.loss.loss(prediction, y)
                initial_gradient = self.loss.gradient(prediction, y)#self.optimiser.gradient(self.loss, prediction, y)
                self.backward(initial_gradient)
            if epoch % loss_step == 0: print(f"Epoch: {epoch} - Error: {error / len(X)}")


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Model(2)
model.add(Dense(3))
model.add(Tanh(3))
model.add(Dense(1))
model.add(Tanh(1))
model.compile(optimiser=None, loss=mse())
model.summary()
model.fit(x, y, epochs=10000, loss_step=500)
print(model.predict(np.array([0, 0])))
print(model.predict(np.array([0, 1])))
print(model.predict(np.array([1, 0])))
print(model.predict(np.array([1, 1])))
print(model.predict(np.array([0.5, 0.5])))
