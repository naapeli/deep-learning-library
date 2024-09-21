from src.DLL.MachineLearning.SupportVectorMachines.SVC import SVC, SVCSMO
from src.DLL.MachineLearning import Kernels
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


X, y = datasets.make_blobs(n_samples=100, n_features=2, cluster_std=1, centers=4, random_state=2)

plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
plt.show()

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model1 = SVC(kernel=Kernels.LinearCovariance())
model1.fit(X_train, y_train)
predictions = model1.predict(X_test)
print(round(accuracy(predictions, y_test), 3))

n = 500
x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions = model1.predict(X_grid).reshape((n, n))

model2 = SVCSMO(kernel=Kernels.LinearCovariance())
model2.fit(X_train, y_train, epochs=1000)
predictions = model2.predict(X_test)
print(round(accuracy(predictions, y_test), 3))

n = 500
x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions2 = model2.predict(X_grid).reshape((n, n))

fig, axes = plt.subplots(1, 2)
axes[0].contourf(Xv.numpy(), Yv.numpy(), grid_predictions.numpy(), alpha=0.5)
axes[1].contourf(Xv.numpy(), Yv.numpy(), grid_predictions2.numpy(), alpha=0.5)
axes[0].scatter(X[:, 0], X[:, 1], c=y, s=5)
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=5)
axes[0].set_title("cvxopt solver")
axes[1].set_title("My SMO solver")
plt.show()
