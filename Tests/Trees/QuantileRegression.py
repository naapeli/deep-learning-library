"""
Quantile regression and prediction intervals
==============================================

This script trains models with the quantile (pinball) loss. Instead of the mean of the target, a model trained
with the quantile loss predicts a chosen quantile :math:`\\tau` of the target given the features. Training one
model for :math:`\\tau = 0.05` and another for :math:`\\tau = 0.95` gives a 90 % prediction interval, whose width
can change with the features. This is not possible with a single model trained with the squared error.

The quantile loss is used with the gradient boosting, XGBoost and LightGBM regressors (``loss="quantile"``) and with
a neural network (:class:`~DLL.DeepLearning.Losses.Quantile`).
"""
import torch
import matplotlib.pyplot as plt

from DLL.MachineLearning.SupervisedLearning.Trees import GradientBoostingRegressor, XGBoostingRegressor, LGBMRegressor
from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Losses import Quantile
from DLL.DeepLearning.Optimisers import ADAM


torch.manual_seed(0)

# %%
# Data
# ----
#
# The noise of the target grows with :math:`x`, so a good prediction interval should widen from left to right.


def generate(n):
    X = torch.rand(n, 1) * 10
    y = 2 * torch.sin(X[:, 0]) + torch.randn(n) * (0.2 + 0.15 * X[:, 0])
    return X, y


X_train, y_train = generate(300)
X_test, y_test = generate(1000)
X_plot = torch.linspace(0, 10, 200).unsqueeze(1)
quantiles = [0.05, 0.5, 0.95]

# %%
# Boosting machines
# -----------------
#
# Every quantile needs its own model. The share of the test targets below the predicted quantile (the coverage)
# should be close to the quantile itself.

models = {
    "Gradient boosting": lambda q: GradientBoostingRegressor(n_trees=40, learning_rate=0.2, max_depth=2, loss="quantile", quantile=q),
    "XGBoost": lambda q: XGBoostingRegressor(n_trees=40, learning_rate=0.2, max_depth=2, loss="quantile", quantile=q),
    "LightGBM": lambda q: LGBMRegressor(n_trees=40, learning_rate=0.2, max_depth=2, loss="quantile", quantile=q),
}
predictions = {}
for name, model_class in models.items():
    predictions[name] = []
    for q in quantiles:
        model = model_class(q)
        model.fit(X_train, y_train)
        coverage = (y_test <= model.predict(X_test)).float().mean().item()
        print(f"{name}, quantile {q}: coverage {coverage:.3f}")
        predictions[name].append(model.predict(X_plot))

# %%
# Neural network
# --------------
#
# A small network trained with the quantile loss for every quantile.

predictions["Neural network"] = []
for q in quantiles:
    torch.manual_seed(0)
    model = Model(1)
    model.add(Dense(32, activation=Tanh()))
    model.add(Dense(32, activation=Tanh()))
    model.add(Dense(0))
    model.compile(optimiser=ADAM(learning_rate=0.01), loss=Quantile(quantile=q))
    model.fit(X_train / 10, y_train, epochs=300, batch_size=64)
    coverage = (y_test <= model.predict(X_test / 10)).float().mean().item()
    print(f"Neural network, quantile {q}: coverage {coverage:.3f}")
    predictions["Neural network"].append(model.predict(X_plot / 10))

# %%
# Prediction intervals
# --------------------
#
# The shaded area is the predicted 90 % interval and the line is the predicted median.

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (name, (lower, median, upper)) in zip(axes.ravel(), predictions.items()):
    ax.scatter(X_train[:, 0], y_train, s=3, alpha=0.4, color="gray", label="training data")
    ax.fill_between(X_plot[:, 0], lower, upper, alpha=0.3, label="90 % interval")
    ax.plot(X_plot[:, 0], median, label="median")
    ax.plot(X_plot[:, 0], 2 * torch.sin(X_plot[:, 0]), "k--", linewidth=1, label="true median")
    ax.set_title(name)
    ax.grid(alpha=0.3)
axes[0, 0].legend()
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
plt.tight_layout()
plt.show()

# %%
# Every model captures the growing noise. The intervals of the boosting machines are slightly too narrow on the
# test data (a coverage of about 7 % and 93 % instead of 5 % and 95 %), because the trees partly fit the noise of
# the training data. Fewer or shallower trees or a smaller learning rate would widen them.
