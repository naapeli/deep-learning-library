"""
Multidimensional Gaussian Process Regression (GPR)
========================================================

This script demonstrates the use of a Gaussian Process Regressor (GPR) with a 
Radial Basis Function (RBF) kernel in a multidimensional setting. The example 
involves training a GPR model on 2D input data and predicting the outputs on 
a test set.
"""

import torch
import matplotlib.pyplot as plt

from DLL.MachineLearning.SupervisedLearning.GaussianProcesses import GaussianProcessRegressor
from DLL.MachineLearning.SupervisedLearning.Kernels import RBF, WhiteGaussian
from DLL.Data.Preprocessing import data_split, StandardScaler
from DLL.DeepLearning.Optimisers import ADAM


n = 30
X, Y = torch.meshgrid(torch.linspace(0, 1, n, dtype=torch.float32), torch.linspace(-1, 1, n, dtype=torch.float32), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)
transformer = StandardScaler()
y_train = transformer.fit_transform(y_train)
y_test = transformer.transform(y_test)


model = GaussianProcessRegressor(RBF(correlation_length=torch.tensor([10.0, 10.0],)) + WhiteGaussian())
model.fit(x_train, y_train)
optimizer = ADAM(0.01)
lml = model.train_kernel(epochs=1000, optimiser=optimizer, callback_frequency=100, verbose=True)["log marginal likelihood"]
mean, _ = model.predict(x_test)
z = transformer.inverse_transform(mean)

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')
surf = ax.scatter(x_test[:, 0], x_test[:, 1], z, color="blue", label="prediction")
surf = ax.scatter(x_test[:, 0], x_test[:, 1], transformer.inverse_transform(y_test), color="red", label="true value")
ax.legend()

ax = fig.add_subplot(122)
ax.plot(lml)
ax.grid()
ax.set_xlabel("Epoch")
ax.set_ylabel("Log marginal likelihood")

plt.show()
