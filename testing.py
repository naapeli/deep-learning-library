from Model import Model
from Layers.Dense import Dense
from Layers.Activations.Tanh import Tanh
from Layers.Activations.ReLU import ReLU
from Losses.MSE import mse
import numpy as np
import matplotlib.pyplot as plt


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 1, 2)
y = np.array([0, 1, 1, 0])
# errors = model.fit(x, y, epochs=1000, loss_step=100)

model = Model(2)
model.add(Dense(3, activation=Tanh()))
model.add(Dense(1, activation=Tanh()))
model.compile(optimiser=None, loss=mse())
model.summary()
errors = model.fit(x.reshape(1, 4, 2), y.reshape(1, 4), epochs=1000, loss_step=100)
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")

Xv, Yv = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
z = model.predict(np.array([Xv.flatten(), Yv.flatten()]).reshape(1, 2, 2500))
z = z.reshape(Xv.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(Xv, Yv, z)
fig.colorbar(surf)
plt.show()
