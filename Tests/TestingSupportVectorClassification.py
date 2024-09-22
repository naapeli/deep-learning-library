import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from time import perf_counter
from sklearn import svm

from src.DLL.MachineLearning.SupervisedLearning.SupportVectorMachines.SVC import SVC, SVCSMO
from src.DLL.MachineLearning.SupervisedLearning import Kernels
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy


X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=2, centers=4, random_state=3)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
# plt.show()

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model1 = SVC(kernel=Kernels.SquaredExponentialCovariance(), C=1000)
start1 = perf_counter()
model1.fit(X_train, y_train, multi_method="ovr")
end1 = perf_counter()
predictions = model1.predict(X_test)
print(round(accuracy(predictions, y_test), 3))

n = 100
x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions = model1.predict(X_grid).reshape((n, n))

model2 = SVCSMO(kernel=Kernels.SquaredExponentialCovariance(), C=1000)
start2 = perf_counter()
model2.fit(X_train, y_train, epochs=1, multi_method="ovr")
end2 = perf_counter()
predictions = model2.predict(X_test)
print(round(accuracy(predictions, y_test), 3))

x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions2 = model2.predict(X_grid).reshape((n, n))

model3 = svm.SVC(kernel="rbf", C=1000, gamma=1, decision_function_shape='ovr')
start3 = perf_counter()
model3.fit(X_train, y_train)
end3 = perf_counter()
predictions = model3.predict(X_test)
print(round(accuracy(predictions, y_test), 3))

x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions3 = model3.predict(X_grid).reshape((n, n))

fig, axes = plt.subplots(1, 3)
axes[0].contourf(Xv.numpy(), Yv.numpy(), grid_predictions.numpy(), alpha=0.5)
axes[1].contourf(Xv.numpy(), Yv.numpy(), grid_predictions2.numpy(), alpha=0.5)
axes[2].contourf(Xv.numpy(), Yv.numpy(), grid_predictions3, alpha=0.5)
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
axes[0].set_title(f"cvxopt solver - time to train: {round(end1 - start1, 3)}")
axes[1].set_title(f"My SMO solver - time to train: {round(end2 - start2, 3)}")
axes[2].set_title(f"SKlearn implementation - time to train: {round(end3 - start3, 3)}")
plt.show()
