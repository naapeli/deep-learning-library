"""
Imbalanced binary classification with logistic regression
===========================================================

This script trains a logistic regression model on a binary dataset, where only 5 % of the samples belong to the
positive class. A model trained on such data is dominated by the many negative samples: it reaches a high accuracy
by predicting the negative class almost everywhere, but it misses most of the positive samples.

A common fix is to weight the samples in the loss with ``sample_weight``. With the *balanced* weights

.. math::

    w_i = \\frac{n}{2 n_{y_i}},

where :math:`n_{y_i}` is the number of samples in the class of sample :math:`i`, both classes contribute equally to
the loss. The positive samples then get a much larger weight than the negative ones, and the decision boundary moves
towards the negative class.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from DLL.MachineLearning.SupervisedLearning.LinearModels import LogisticRegression
from DLL.Data.Preprocessing import data_split, StandardScaler
from DLL.Data.Metrics import accuracy, precision, recall, roc_curve, auc


torch.manual_seed(0)

# %%
# Data
# ----
#
# Two features, so that the decision boundaries can be plotted, and a 95 % / 5 % split between the classes. The
# classes overlap, so no linear model can separate them perfectly.

X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, weights=[0.95, 0.05], class_sep=1.0, flip_y=0, random_state=4)
X = StandardScaler().fit_transform(torch.tensor(X, dtype=torch.float32))
y = torch.tensor(y, dtype=torch.float32)
X_train, y_train, _, _, X_test, y_test = data_split(X, y, train_split=0.7, validation_split=0.0)
print(f"Positive samples: {int(y_train.sum())} of {len(y_train)} in the training data and {int(y_test.sum())} of {len(y_test)} in the test data")

# %%
# Sample weights
# --------------
#
# The balanced weights give every class the same total weight. Any other weights can be used as well, for instance
# to make missing a positive sample a chosen number of times more costly than a false alarm.

n_positive = y_train.sum()
n_negative = len(y_train) - n_positive
balanced_weights = torch.where(y_train == 1, len(y_train) / (2 * n_positive), len(y_train) / (2 * n_negative))
print(f"Weight of a positive sample: {balanced_weights[y_train == 1][0]:.2f}, weight of a negative sample: {balanced_weights[y_train == 0][0]:.2f}")

# %%
# Training
# --------
#
# The same model is trained without weights and with the balanced weights.

models = {}
for name, weights in [("unweighted", None), ("balanced weights", balanced_weights)]:
    torch.manual_seed(0)
    model = LogisticRegression(learning_rate=0.01)
    model.fit(X_train, y_train, sample_weight=weights, epochs=1000)
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    fpr, tpr = roc_curve(probabilities, y_test, torch.linspace(0, 1, 500))
    print(f"{name}: accuracy {accuracy(predictions, y_test):.3f}, precision {precision(predictions, y_test):.3f}, recall {recall(predictions, y_test):.3f}, ROC AUC {auc(fpr, tpr):.3f}")
    models[name] = (model, fpr, tpr)

# %%
# The unweighted model has the higher accuracy, because it is right about the negative samples, which are almost
# all of the data. The weighted model finds most of the positive samples (a much higher recall) at the cost of more
# false alarms (a lower precision). Which one is better depends on the cost of a missed positive sample compared to
# the cost of a false alarm.
#
# Decision boundaries and ROC curves
# ----------------------------------
#
# The lines are the decision boundaries at the probability 0.5. The weights move the boundary towards the negative
# class. The ROC curves are almost the same, since the weights mostly shift the predicted probabilities instead of
# changing how the model ranks the samples. The unweighted model can find the same positive samples as the weighted
# one by using a lower threshold than 0.5.

n = 200
x_grid = torch.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, n)
y_grid = torch.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, n)
Xv, Yv = torch.meshgrid(x_grid, y_grid, indexing="ij")
X_grid = torch.stack((Xv.flatten(), Yv.flatten()), dim=1)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], s=5, alpha=0.4, color="gray", label="negative class")
ax[0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], s=12, color="red", label="positive class")
for (name, (model, fpr, tpr)), color in zip(models.items(), ["C0", "C2"]):
    probabilities = model.predict_proba(X_grid).reshape(n, n)
    ax[0].contour(Xv.numpy(), Yv.numpy(), probabilities.numpy(), levels=[0.5], colors=color)
    ax[0].plot([], [], color=color, label=f"{name} boundary")
    ax[1].plot(fpr, tpr, color=color, label=f"{name}, AUC = {auc(fpr, tpr):.3f}")
ax[0].set_xlabel("feature 1")
ax[0].set_ylabel("feature 2")
ax[0].set_title("Test data and decision boundaries")
ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
ax[1].set_xlabel("false positive rate")
ax[1].set_ylabel("true positive rate")
ax[1].set_title("ROC curves")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()
