"""
Kolmogorov-Arnold Networks
==================================

This script implements a model using Kolmogorov-Arnold networks. It fits to a simple 
quadratic surface using only a few parameters.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Layers._DenseKAN import _get_basis_functions, _NeuronKAN
from DLL.DeepLearning.Layers import DenseKAN, Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import MSE
from DLL.DeepLearning.Initialisers import Xavier_Normal
from DLL.Data.Preprocessing import data_split


# X = torch.linspace(-1, 1, 100).unsqueeze(1)
# y = torch.sin(4 * X).squeeze()
# X = torch.linspace(-1, 1, 50).unsqueeze(1)
# y = (0.5 * torch.sin(4 * X) * torch.exp(-X - 1) + 0.5).squeeze()
n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
X, y, _, _, X_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)


model = Model(2)
# model.add(DenseKAN(1, activation=Tanh(), initialiser=Xavier_Normal()))
model.add(DenseKAN(2, activation=Tanh(), initialiser=Xavier_Normal()))
# model.add(DenseKAN(2, activation=Tanh(), initialiser=Xavier_Normal()))
model.add(DenseKAN(2, initialiser=Xavier_Normal()))
model.add(Dense(0, initialiser=Xavier_Normal()))
model.compile(ADAM(0.01), MSE())
model.summary()

history = model.fit(X, y, epochs=300, verbose=True)

# X_test = 2 * torch.rand_like(X) - 1
# y_test = torch.sin(4 * X_test).squeeze()
# X_test = 2 * torch.rand_like(X) - 1
# y_test = (0.5 * torch.sin(4 * X_test) * torch.exp(-X_test - 1) + 0.5).squeeze()

# plt.figure()
# plt.scatter(X_test, y_test, label="True test points")
# plt.scatter(X_test, model.predict(X_test), label="Predictions")
# plt.scatter(X, y, label="True train points")
# plt.scatter(X, model.predict(X), label="Predicted train points")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="True test points")
ax.scatter(X_test[:, 0], X_test[:, 1], model.predict(X_test), label="Predictions")
plt.legend()


plt.figure(figsize=(8, 8))
plt.plot(history["loss"])
plt.title("Loss function as a function of epoch")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.show()

n_fun = 10
basis_funcs, basis_func_derivatives = _get_basis_functions(n_fun, degree=5, bounds=(-1, 1))
x = torch.linspace(-1, 1, 100).unsqueeze(1)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
plt.subplots_adjust(wspace=0.3)
for i in range(n_fun):
    basis_values = basis_funcs[i](x)
    derivative_values = basis_func_derivatives[i](x)

    ax[0].plot(x.squeeze(1), basis_values)
    ax[1].plot(x, derivative_values)

ax[0].set_title("B-spline Basis Functions")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].grid()
ax[1].set_title("B-spline Basis Function Derivatives")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].grid()
plt.show()
