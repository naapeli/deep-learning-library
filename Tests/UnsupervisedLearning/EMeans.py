"""
e-means versus k-means
=========================

This script compares :class:`~DLL.MachineLearning.UnsupervisedLearning.Clustering.EMeansClustering` with
:class:`~DLL.MachineLearning.UnsupervisedLearning.Clustering.KMeansClustering` (Lloyd's algorithm).

e-means is mini-batch k-means where every centroid is an exponential moving average of the points assigned
to it. Its batch size and step size follow scaling laws fitted in the paper (S. Bessa et al., "e-means:
outperforming k-means (in one-tenth the epochs)", 2026), so the only choice left is the number of epochs.

The models are fitted on synthetic "embedding-like" data (many overlapping blobs plus a shared low-rank
component and noise) and evaluated by the mean squared distance of held-out points to their nearest centroid.
A pass over the data is one Lloyd iteration for k-means and one epoch for e-means.
"""
import time

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.datasets import make_blobs

from DLL.MachineLearning.UnsupervisedLearning.Clustering import KMeansClustering, EMeansClustering

EMEANS_COLOR, KMEANSPP_COLOR, RANDOM_COLOR = "#2a78d6", "#eb6834", "#1baf7a"


def make_data(n, d, n_blobs=3000, rank=16, seed=1):
    generator = torch.Generator().manual_seed(seed)
    low_rank = torch.randn(n, rank, generator=generator) @ torch.randn(rank, d, generator=generator) * 0.5
    centers = torch.randn(n_blobs, d, generator=generator)
    blob = torch.randint(0, n_blobs, (n,), generator=generator)
    return centers[blob] + low_rank + 0.3 * torch.randn(n, d, generator=generator)


def held_out_mse(model, X):
    return ((X - model.centroids[model.predict(X)]) ** 2).sum(dim=1).mean().item()


def fit(model, X, seed):
    torch.manual_seed(seed)
    start = time.perf_counter()
    model.fit(X)
    return time.perf_counter() - start


n_train, n_test, d, k = 30_000, 10_000, 32, 128
data = make_data(n_train + n_test, d)
X_train, X_test = data[:n_train], data[n_train:]

# %%
# Quality per pass over the data and per second
# ----------------------------------------------
# k-means is stopped after 1, 2, ..., 25 iterations (``tol=0`` disables the early stopping) and e-means is
# fitted with a budget of 1, 2, 3, 5 and 8 epochs. Each e-means run is a separate fit, since the step
# schedule depends on the epoch budget.

iterations = list(range(1, 26))
epochs = [1, 2, 3, 5, 8]
curves = {}
for init in ["random", "kmeans++"]:
    curves[init] = {"mse": [], "time": []}
    for n_iter in iterations:
        model = KMeansClustering(k=k, init=init, n_init=1, max_iters=n_iter, tol=0)
        curves[init]["time"].append(fit(model, X_train, seed=0))
        curves[init]["mse"].append(held_out_mse(model, X_test))
curves["e-means"] = {"mse": [], "time": []}
for n_epochs in epochs:
    model = EMeansClustering(k=k, n_epochs=n_epochs)
    curves["e-means"]["time"].append(fit(model, X_train, seed=0))
    curves["e-means"]["mse"].append(held_out_mse(model, X_test))
    print(f"e-means, {n_epochs} epochs: held-out MSE {curves['e-means']['mse'][-1]:.3f}, "
          f"batch size {model.resolved_batch_size}, alpha0 {model.resolved_alpha0:.3f}, {model.n_revived} revived clusters")
print(f"k-means++, 25 iterations: held-out MSE {curves['kmeans++']['mse'][-1]:.3f}")

series = [
    ("random", "k-means (random init)", RANDOM_COLOR, "s", iterations),
    ("kmeans++", "k-means (k-means++ init)", KMEANSPP_COLOR, "^", iterations),
    ("e-means", "e-means (1, 2, 3, 5, 8 epochs)", EMEANS_COLOR, "o", epochs),
]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for key, label, color, marker, passes in series:
    ax[0].plot(passes, curves[key]["mse"], color=color, marker=marker, markersize=5, label=label)
    ax[1].plot(curves[key]["time"], curves[key]["mse"], color=color, marker=marker, markersize=5, label=label)
for n_epochs, mse in zip(epochs, curves["e-means"]["mse"]):
    ax[0].annotate(f"{n_epochs}", (n_epochs, mse), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
low = min(min(c["mse"]) for c in curves.values())
for a in ax:
    a.set_xscale("log")
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(low - 0.3, low + 3.5)
    a.set_ylabel("Held-out MSE")
    a.grid(alpha=0.3)
    a.legend()
ax[0].set_xlabel("Passes over the data")
ax[0].set_title("Quality per pass over the data")
ax[1].set_xlabel("Fit time [s]")
ax[1].set_title("Quality per second")
fig.suptitle(f"n = {n_train}, d = {d}, k = {k}")
plt.tight_layout()

# %%
# Variation between random seeds
# --------------------------------
# Both algorithms depend on the random initialisation. The held-out error of 3 epochs of e-means is compared
# with 25 iterations of k-means over 5 seeds. The horizontal line is the median.

seeds = range(5)
results = {
    "e-means\n3 epochs": (EMEANS_COLOR, "o", lambda: EMeansClustering(k=k, n_epochs=3)),
    "k-means++\n25 iterations": (KMEANSPP_COLOR, "^", lambda: KMeansClustering(k=k, n_init=1, max_iters=25, tol=0)),
    "k-means random\n25 iterations": (RANDOM_COLOR, "s", lambda: KMeansClustering(k=k, init="random", n_init=1, max_iters=25, tol=0)),
}
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (label, (color, marker, make_model)) in enumerate(results.items()):
    errors = []
    for seed in seeds:
        model = make_model()
        fit(model, X_train, seed)
        errors.append(held_out_mse(model, X_test))
    errors = torch.tensor(errors)
    ax.scatter(i + torch.linspace(-0.1, 0.1, len(errors)), errors, color=color, marker=marker, s=50, zorder=3)
    ax.hlines(errors.median(), i - 0.25, i + 0.25, color="black", linewidth=1.5)
ax.set_xticks(range(len(results)), list(results.keys()))
ax.set_xlim(-0.5, len(results) - 0.5)
ax.set_ylabel("Held-out MSE")
ax.set_title("Spread over 5 seeds")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

# %%
# The e-means step schedule
# ---------------------------
# The EMA step :math:`\alpha = 1 - \gamma` follows a cosine schedule from :math:`\alpha_0` to
# :math:`0.1 \alpha_0`. Larger epoch budgets use larger batches, so the number of steps grows slower than
# the number of epochs.

fig, ax = plt.subplots(figsize=(7, 4.5))
for n_epochs, color in zip([1, 3, 8], ["#86b6ef", "#2a78d6", "#104281"]):
    batch_size = EMeansClustering.batch_size_law(n_train, k, n_epochs, d)
    alpha0 = EMeansClustering.step_law(n_train, k, n_epochs, batch_size)
    total_steps = n_epochs * -(-n_train // batch_size)
    steps = torch.arange(total_steps)
    alphas = [EMeansClustering._cosine_step(alpha0, t, total_steps) for t in steps.tolist()]
    ax.plot(steps / total_steps * n_epochs, alphas, color=color,
            label=f"{n_epochs} epoch{'s' if n_epochs > 1 else ''}: batch size {batch_size}, {total_steps} steps")
ax.set_xlabel("Epoch")
ax.set_ylabel(r"EMA step $\alpha$")
ax.set_title("Step schedule chosen by the scaling laws")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()

# %%
# Two-dimensional data
# ----------------------
# In very low dimensions, e-means uses small batches and no dead-cluster revival. A poorly placed
# centroid is then not moved elsewhere, and a careful initialisation such as k-means++ can win.

X_2d, _ = make_blobs(n_samples=6000, centers=15, cluster_std=0.7, random_state=4)
X_2d = torch.from_numpy(X_2d)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
models = [("k-means++", KMeansClustering(k=15, n_init=1)), ("e-means, 3 epochs", EMeansClustering(k=15, n_epochs=3))]
for a, (name, model) in zip(ax, models):
    fit(model, X_2d, seed=0)
    a.scatter(X_2d[:, 0], X_2d[:, 1], c=model.labels, cmap="tab20", s=3, alpha=0.6)
    a.scatter(model.centroids[:, 0], model.centroids[:, 1], marker="X", s=120, c="black", edgecolors="white")
    a.set_title(f"{name}: MSE {model.inertia / len(X_2d):.3f}")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
