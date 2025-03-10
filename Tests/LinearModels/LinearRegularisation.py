import torch
import matplotlib.pyplot as plt
from math import log10

from src.DLL.MachineLearning.SupervisedLearning.LinearModels import LASSORegression, RidgeRegression, ElasticNetRegression
from src.DLL.DeepLearning.Optimisers import ADAM


n = 10
x1 = torch.linspace(0, 1, n)
x2 = torch.linspace(0, 1, n)
x3 = torch.linspace(0, 1, n)
XX1, XX2, XX3 = torch.meshgrid(x1, x2, x3, indexing="xy")
X = torch.stack((XX1.flatten(), XX2.flatten(), XX3.flatten()), dim=1)
y = 2 * XX1.flatten() - 5 * XX2.flatten() + 1 * XX3.flatten() + 0.1 * torch.normal(0, 1, size=XX1.flatten().size())

weights = []
alphas = torch.logspace(log10(1e-1), log10(1e5), 20).tolist()
for alpha in alphas:
    model = RidgeRegression(alpha=alpha)
    model.fit(X, y)
    weights.append(model.beta.tolist()[1:])

weights = torch.tensor(weights)

fig, axes = plt.subplots(1, 2)

for row in weights.T:
    axes[0].semilogx(alphas, row)
axes[0].set_title("Ridge regression")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Weights")

weights = []
alphas = torch.logspace(log10(1e-3), log10(2e0), 20).tolist()
LASSO = False
for alpha in alphas:
    if LASSO:
        model = LASSORegression(alpha=alpha)
    else:
        model = ElasticNetRegression(alpha=alpha, l1_ratio=0.5)
    print(alpha)
    model.fit(X, y, epochs=10000, optimiser=ADAM(learning_rate=0.001))
    weights.append([model.weights.tolist()])

alphas = torch.logspace(log10(1e-3), log10(2e0), 20).tolist()
weights = torch.tensor(weights).squeeze()

for row in weights.T:
    axes[1].semilogx(alphas, row)
title = f"{'LASSO' if LASSO else 'ElasticNet'} regression"
axes[1].set_title(title)
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Weights")
plt.show()
