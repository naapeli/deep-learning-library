"""
Gaussian Process Classification (GPC)
===================================

This script demonstrates the use of a Gaussian Process Classification (GPC) 
model with a the RBF kernel on generated data. The model is trained using 
the Adam optimizer to obtain optimal hyperparameters. The final model is
compared against the untrained model.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, make_moons
from copy import deepcopy

from DLL.MachineLearning.SupervisedLearning.GaussianProcesses import GaussianProcessClassifier
from DLL.MachineLearning.SupervisedLearning.Kernels import RBF, Linear, Matern, RationalQuadratic
from DLL.Data.Preprocessing import data_split
from DLL.Data.Metrics import accuracy
from DLL.DeepLearning.Optimisers import ADAM


n_samples = 100
dataset = "moons"
if dataset == "basic": X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1, centers=2, random_state=3)
if dataset == "narrow": X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=4)
if dataset == "moons": X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
X = torch.from_numpy(X)
y = torch.from_numpy(y).to(X.dtype)

X_train, y_train, _, _, X_test, y_test = data_split(X, y, train_split=0.7, validation_split=0.0)

# untrained_model = GaussianProcessClassifier(RBF(correlation_length=torch.tensor([1.0, 1.0])), n_iter_laplace_mode=50)  # anisotropic kernel (each coordinate has different length scale)
untrained_model = GaussianProcessClassifier(RBF(correlation_length=3), n_iter_laplace_mode=50)  # isotropic kernel (each coordinate has the same length scale)
model = deepcopy(untrained_model)
untrained_model.fit(X_train, y_train)
model.fit(X_train, y_train)

optimizer = ADAM(0.1)  # make the learning rate a little larger than default as 0.001 takes a long time to converge. Could increase even more if wanted the true optimum for the sigma as well.
lml = model.train_kernel(epochs=200, optimiser=optimizer, callback_frequency=10, verbose=True)["log marginal likelihood"]

y_pred = model.predict(X_test)
print("Trained model test accuracy:", accuracy(y_pred, y_test))
y_pred = untrained_model.predict(X_test)
print("Untrained model test accuracy:", accuracy(y_pred, y_test))


x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
n = 75
xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, n, dtype=X.dtype),
                        torch.linspace(y_min, y_max, n, dtype=X.dtype), indexing='ij')
grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)

with torch.no_grad():
    proba_untrained = untrained_model.predict_proba(grid).reshape(xx.shape)
    proba_trained = model.predict_proba(grid).reshape(xx.shape)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
blue = "#3366cc"
red = "#cc3333"
for ax, proba, title in zip(
    axs, [proba_untrained, proba_trained], ["Untrained", "Trained"]
):
    contour = ax.contourf(xx.numpy(), yy.numpy(), proba.numpy(), levels=100, cmap="bwr", alpha=0.7, vmin=0, vmax=1)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c=blue, edgecolor='k', label='Train 0')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c=red, edgecolor='k', label='Train 1')
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c=blue, edgecolor='k', marker='^', label='Test 0')
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c=red, edgecolor='k', marker='^', label='Test 1')
    ax.set_title(f"{title} Model")
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.legend()

fig.colorbar(contour, ax=axs.ravel().tolist(), label="$\\mathbb{P}(y = 1)$")

plt.figure(figsize=(8, 8))
plt.plot(lml.numpy())
plt.title("Log marginal likelihood as a function of training iterations")
plt.ylabel("Log marginal likelihood")
plt.xlabel("Epochs")
plt.grid()

plt.show()
