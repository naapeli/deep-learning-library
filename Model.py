from Layers.Dense import Dense
from Layers.Activations.Tanh import Tanh
from Layers.Input import Input
from Losses.MSE import mse
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, input_size):
        self.layers = [Input(input_size)]
        self.optimiser = None
        self.loss = mse()

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

    def fit(self, X, Y, val_X=None, val_Y=None, epochs=100, loss_step=5):
        errors = np.zeros(epochs)
        for epoch in range(epochs):
            # calculate the loss
            error = 0
            for x, y in zip(X, Y):
                prediction = self.predict(x)
                error += self.loss.loss(prediction, y)
                initial_gradient = self.loss.gradient(prediction, y)
                # self.optimiser.gradient(initial_gradient)
                self.backward(initial_gradient)
            error /= len(X)
            errors[epoch] = error
            if epoch % loss_step == 0: print(f"Epoch: {epoch} - Error: {error}")
        return errors


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Model(2)
model.add(Dense(3, activation=Tanh()))
model.add(Dense(1, activation=Tanh()))
model.compile(optimiser=None, loss=mse())
model.summary()
errors = model.fit(x, y, epochs=1000, loss_step=100)
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
plt.show()
print(model.predict(np.array([0, 0])))
print(model.predict(np.array([0, 1])))
print(model.predict(np.array([1, 0])))
print(model.predict(np.array([1, 1])))
print(model.predict(np.array([0.5, 0.5])))

Xv, Yv = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
values = []
for x, y in zip(Xv.flatten(), Yv.flatten()):
    values.append(model.predict(np.array([x, y])))
z = np.array(values)
z = z.reshape(Xv.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(Xv, Yv, z, cmap='viridis')
fig.colorbar(surf)
plt.show()
