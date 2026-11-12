"""
Support Vector Classifier Solver Comparison
===============================================

This script compares the performance of different Support Vector Machine (SVM) solvers on a synthetic 2D classification dataset. The solvers compared include:

1. **CVXOPT Solver**: Custom SVM optimization using the CVXOPT library.
2. **SMO Solver**: Custom implementation of the Sequential Minimal Optimization (SMO) algorithm.
3. **Coordinate Ascent Solver**: Custom implementation of the coordinate ascent optimization method for SVM.
4. **Gradient Descent Solver**: Minimises the hinge loss or the squared hinge loss (:class:`~DLL.DeepLearning.Losses.Hinge`) directly in the primal form.
5. **Scikit-learn SVM**: Standard implementation from the Scikit-learn library.

The performance is evaluated based on prediction accuracy and training time. Visualizations of decision boundaries learned by each solver on the test set are provided for comparison.
"""
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from time import perf_counter
from sklearn import svm

from DLL.MachineLearning.SupervisedLearning.SupportVectorMachines import SVC
from DLL.MachineLearning.SupervisedLearning import Kernels
from DLL.Data.Preprocessing import data_split
from DLL.Data.Metrics import accuracy, prob_to_pred
from DLL.DeepLearning.Losses import Hinge

# %%
# Data
# ----
#
# Four overlapping Gaussian blobs in two dimensions. Every model uses an RBF kernel with a length scale of 5 and
# C = 1, and the decision regions are evaluated on a 100 x 100 grid.

X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=2, centers=4, random_state=3)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
# plt.show()

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

# %%
# CVXOPT solver
# -------------
#
# The dual problem is solved as a quadratic program with CVXOPT. The four classes are handled with one-vs-one
# classifiers.

model1 = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="cvxopt")
start1 = perf_counter()
model1.fit(X_train, y_train, epochs=2, multi_method="ovo")
end1 = perf_counter()
predictions = prob_to_pred(model1.predict_proba(X_test))
print(round(accuracy(predictions, y_test), 3))

n = 100
x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions = model1.predict(X_grid).reshape((n, n))

# %%
# SMO solver
# ----------
#
# Sequential minimal optimisation, also one-vs-one.

model2 = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="smo")
start2 = perf_counter()
model2.fit(X_train, y_train, epochs=5, multi_method="ovo")
end2 = perf_counter()
predictions = prob_to_pred(model2.predict_proba(X_test))
print(round(accuracy(predictions, y_test), 3))

x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions2 = model2.predict(X_grid).reshape((n, n))

# %%
# Coordinate ascent solver
# ------------------------
#
# Coordinate ascent on the dual problem, with one-vs-rest classifiers.

model3 = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="coord_ascent")
start3 = perf_counter()
model3.fit(X_train, y_train, epochs=5, multi_method="ovr")
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

# %%
# Gradient descent solver
# -----------------------
#
# The other solvers solve the dual problem. The gradient descent solver instead minimises the primal objective
# :math:`\frac{1}{2}\|f\|^2 + C\sum_i l(y_i, f(x_i))` directly, where :math:`l` is the hinge loss. The gradient
# of the hinge loss comes from :class:`~DLL.DeepLearning.Losses.Hinge`, and the step size decreases as in Pegasos.
# With ``loss="squared_hinge"``, the loss is differentiable and has a nonzero hessian, so Newton's method is used.

model5 = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="gradient_descent", loss="hinge")
start5 = perf_counter()
model5.fit(X_train, y_train, epochs=300, multi_method="ovr")
end5 = perf_counter()
print(round(accuracy(model5.predict(X_test), y_test), 3))
grid_predictions5 = model5.predict(X_grid).reshape((n, n))

model6 = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="gradient_descent", loss="squared_hinge")
start6 = perf_counter()
model6.fit(X_train, y_train, epochs=50, multi_method="ovr")
end6 = perf_counter()
print(round(accuracy(model6.predict(X_test), y_test), 3))
grid_predictions6 = model6.predict(X_grid).reshape((n, n))

# %%
# Scikit-learn
# ------------
#
# Scikit-learn's SVC with the same kernel for reference.

model4 = svm.SVC(kernel="rbf", C=1, gamma=1/(2 * 5 ** 2), decision_function_shape='ovr')
start4 = perf_counter()
model4.fit(X_train, y_train)
end4 = perf_counter()
predictions = model4.predict(X_test)
print(round(accuracy(torch.from_numpy(predictions), y_test), 3))

x_min, X_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid = torch.linspace(x_min, X_max, n)
y_grid = torch.linspace(y_min, y_max, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")

X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1).to(X.dtype)
grid_predictions4 = model4.predict(X_grid).reshape((n, n))

# %%
# Decision regions
# ----------------

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
results = [
    (grid_predictions, f"cvxopt solver - time to train: {round(end1 - start1, 3)}"),
    (grid_predictions3, f"My coordinate ascent solver - time to train: {round(end3 - start3, 3)}"),
    (grid_predictions2, f"My SMO solver - time to train: {round(end2 - start2, 3)}"),
    (torch.from_numpy(grid_predictions4), f"SKlearn implementation - time to train: {round(end4 - start4, 3)}"),
    (grid_predictions5, f"Hinge loss gradient descent - time to train: {round(end5 - start5, 3)}"),
    (grid_predictions6, f"Squared hinge loss Newton - time to train: {round(end6 - start6, 3)}"),
]
for ax, (grid, title) in zip(axes.ravel(), results):
    ax.contourf(Xv.numpy(), Yv.numpy(), grid.numpy(), alpha=0.5)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)
    ax.set_title(title, fontsize=10)
plt.tight_layout()

# %%
# Convergence of the primal objective
# -----------------------------------
#
# A binary problem between the first two classes. The gradient descent solver stores the primal objective after
# every epoch in ``model.history``. The hinge loss needs many small subgradient steps, while Newton's method
# converges in a few iterations for the squared hinge loss. The objective of the dual solvers is evaluated with the
# hinge loss for comparison.

binary = y_train < 2
X_binary, y_binary = X_train[binary], y_train[binary]
hinge = Hinge(reduction="sum")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for loss, epochs in [("hinge", 300), ("squared_hinge", 50)]:
    model = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method="gradient_descent", loss=loss)
    model.fit(X_binary, y_binary, epochs=epochs)
    ax[0 if loss == "hinge" else 1].plot(model.history["loss"], label=f"gradient descent ({loss})")
for opt_method in ["cvxopt", "coord_ascent"]:
    model = SVC(kernel=Kernels.RBF(correlation_length=5), C=1, opt_method=opt_method)
    model.fit(X_binary, y_binary, epochs=5)
    beta = model.alpha * model.y
    scores = beta @ model._kernel_matrix(X_binary, X_binary) + model.b
    objective = 0.5 * beta @ model._kernel_matrix(X_binary, X_binary) @ beta + hinge.loss(scores, y_binary.to(scores.dtype))
    ax[0].axhline(objective.item(), linestyle="--", color="C1" if opt_method == "cvxopt" else "C2", label=f"{opt_method} (hinge)")
ax[0].set_yscale("log")
for a in ax:
    a.set_xlabel("epoch")
    a.set_ylabel("primal objective")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
plt.show()
