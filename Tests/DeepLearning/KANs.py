"""
Kolmogorov-Arnold Networks
==================================

This script implements a model using Kolmogorov-Arnold networks. It fits to a simple 
quadratic surface using only a few parameters.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Layers._DenseKAN import _make_knots, _evaluate_basis
from DLL.DeepLearning.Layers import DenseKAN, Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import MSE
from DLL.DeepLearning.Initialisers import Xavier_Normal
from DLL.Data.Preprocessing import data_split

# %%
# Data
# ----
#
# Noisy samples of the surface :math:`z = x^2 + y^2 - 5` on a 30 x 30 grid, of which 80 % are used for training.

n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
X, y, _, _, X_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)

# %%
# The model
# ---------
#
# Two ``DenseKAN`` layers followed by a dense output layer. Each ``DenseKAN`` layer stores all of its edge
# functions in a single weight tensor of shape (output_dim, input_dim, n_basis_funcs), so a forward pass is a
# single matrix multiplication.

model = Model(2)
model.add(DenseKAN(2, activation=Tanh(), initialiser=Xavier_Normal()))
model.add(DenseKAN(2, initialiser=Xavier_Normal()))
model.add(Dense(0, initialiser=Xavier_Normal()))
model.compile(ADAM(0.01), MSE())
model.summary()

# %%
# Training
# --------

history = model.fit(X, y, epochs=300, verbose=True)

# %%
# Predictions
# -----------

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="True test points")
ax.scatter(X_test[:, 0], X_test[:, 1], model.predict(X_test), label="Predictions")
plt.legend()

# %%
# Learning curve
# --------------

plt.figure(figsize=(8, 8))
plt.plot(history["loss"])
plt.title("Loss function as a function of epoch")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.show()

# %%
# Basis functions
# ---------------
#
# Every connection of a ``DenseKAN`` layer is a learned combination of a SiLU function and B-spline basis
# functions. Here are 10 basis functions (SiLU and 9 B-splines of degree 5) on [-1, 1] and their derivatives.
# All basis functions are evaluated at once.

n_fun, degree = 10, 5
knots = _make_knots(n_fun - 1, degree, bounds=(-1, 1))
x = torch.linspace(-1, 1, 200)
basis_values, derivative_values = _evaluate_basis(x, knots, degree, derivative=True)  # (200, n_fun)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
plt.subplots_adjust(wspace=0.3)
ax[0].plot(x, basis_values)
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
