"""
Support Vector Classifier
============================

This script trains support vector classifiers on a synthetic 2D classification dataset and compares them to the
implementation of scikit-learn. The models minimise the primal objective of the support vector machine with
Newton's method using either

1. the **hinge loss** (:class:`~DLL.DeepLearning.Losses.Hinge`), which gives the classical support vector machine, or
2. the **squared hinge loss**, which penalises large violations of the margin more.

The hinge loss is not twice differentiable, so its corner is smoothed by a quadratic function during the training,
and the smoothing is decreased towards zero. The squared hinge loss is minimised directly.
"""
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from time import perf_counter
from sklearn import svm

from DLL.MachineLearning.SupervisedLearning.SupportVectorMachines import SVC
from DLL.MachineLearning.SupervisedLearning import Kernels
from DLL.Data.Preprocessing import data_split
from DLL.Data.Metrics import accuracy
from DLL.DeepLearning.Losses import Hinge

# %%
# Data
# ----
#
# Four overlapping Gaussian blobs in two dimensions. Every model uses an RBF kernel with a length scale of 5 and
# C = 1, and the decision regions are evaluated on a 100 x 100 grid.

X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=2, centers=4, random_state=3)
X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

n = 100
x_grid = torch.linspace(X[:, 0].min(), X[:, 0].max(), n)
y_grid = torch.linspace(X[:, 1].min(), X[:, 1].max(), n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")
X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)

# %%
# Fitting the models
# ------------------
#
# The four classes are handled with one-vs-rest classifiers for the hinge loss and one-vs-one classifiers for the
# squared hinge loss.

results = []
for loss, multi_method in [("hinge", "ovr"), ("squared_hinge", "ovo")]:
    model = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, loss=loss)
    start = perf_counter()
    model.fit(X_train, y_train, multi_method=multi_method)
    end = perf_counter()
    print(f"{loss} accuracy: {accuracy(model.predict(X_test), y_test):.3f}")
    results.append((model.predict(X_grid).reshape((n, n)), f"{loss} ({multi_method}) - time to train: {end - start:.3f}"))

model = svm.SVC(kernel="rbf", C=1, gamma=1 / (2 * 5 ** 2), decision_function_shape="ovr")
start = perf_counter()
model.fit(X_train, y_train)
end = perf_counter()
print(f"scikit-learn accuracy: {accuracy(torch.from_numpy(model.predict(X_test)), y_test):.3f}")
results.append((torch.from_numpy(model.predict(X_grid)).reshape((n, n)), f"scikit-learn - time to train: {end - start:.3f}"))

# %%
# Decision regions
# ----------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (grid, title) in zip(axes, results):
    ax.contourf(Xv.numpy(), Yv.numpy(), grid.numpy(), alpha=0.5)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
    ax.set_title(title, fontsize=10)
plt.tight_layout()

# %%
# Convergence of the primal objective
# -----------------------------------
#
# A binary problem between the first two classes. The classifier stores the primal objective after every Newton
# iteration in ``model.history``. The steps of the hinge loss come from the four stages of decreasing smoothing,
# while the squared hinge loss converges in a few iterations. The dashed line is the objective of the solution of
# scikit-learn evaluated with the hinge loss.

binary = y_train < 2
X_binary, y_binary = X_train[binary], y_train[binary].to(X_train.dtype)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, loss in enumerate(["hinge", "squared_hinge"]):
    model = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, loss=loss)
    model.fit(X_binary, y_binary.to(torch.int64))
    ax[i].plot(model.history["loss"], marker=".", label=loss)

reference = svm.SVC(kernel="rbf", C=1, gamma=1 / (2 * 5 ** 2)).fit(X_binary, y_binary)
K = Kernels.RBF(correlation_length=5)(X_binary, X_binary).to(X_binary.dtype)
coefficients = torch.zeros(len(X_binary), dtype=X_binary.dtype)
coefficients[reference.support_] = torch.from_numpy(reference.dual_coef_[0])
scores = K @ coefficients + reference.intercept_[0]
objective = 0.5 * coefficients @ K @ coefficients + Hinge(reduction="sum").loss(scores, y_binary)
ax[0].axhline(objective.item(), linestyle="--", color="gray", label="scikit-learn (hinge)")
ax[0].set_yscale("log")
for a in ax:
    a.set_xlabel("Newton iteration")
    a.set_ylabel("primal objective")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
plt.show()
