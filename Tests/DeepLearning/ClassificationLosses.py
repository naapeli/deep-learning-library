"""
Classification losses
=======================

This script compares the classification losses of the library. It shows how the losses behave as a function of
the prediction, trains networks with the binary cross entropy and the focal loss on an imbalanced dataset, and
trains networks on the digits dataset with the categorical cross entropy with hard and soft (smoothed) labels and
with the multi-class hinge loss.

Every network outputs logits, i.e. the last layer has no sigmoid or softmax activation. The losses are constructed
with ``from_logits=True``, which applies the sigmoid or the softmax inside the loss. This is numerically more
stable than a separate activation layer, and the gradient with respect to the logits is simply the difference
between the predicted probabilities and the labels.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import ReLU
from DLL.DeepLearning.Losses import BCE, CCE, Focal, Hinge
from DLL.DeepLearning.Optimisers import ADAM
from DLL.Data.Preprocessing import data_split, OneHotEncoder, StandardScaler
from DLL.Data.Metrics import accuracy, precision, recall, roc_curve, auc


torch.manual_seed(0)

# %%
# The losses as functions of the prediction
# -----------------------------------------
#
# The loss of a single positive sample (label 1) as a function of the logit :math:`z` of the model. For the hinge
# losses, :math:`z` is the score of the classifier. The focal loss multiplies the cross entropy by
# :math:`(1 - p)^\gamma`, where :math:`p = \sigma(z)` is the predicted probability of the true class. The larger
# :math:`\gamma`, the less the well classified samples (large :math:`z`) contribute to the loss and the gradient.
# The hinge loss is exactly zero once the sample is classified correctly with a margin of 1.

z = torch.linspace(-4, 4, 201)
ones = torch.ones(1)


def elementwise(loss, values):
    return torch.stack([loss(value.unsqueeze(0), ones) for value in values])


losses = {
    "cross entropy": BCE(from_logits=True),
    r"focal, $\gamma = 1$": Focal(gamma=1, from_logits=True),
    r"focal, $\gamma = 2$": Focal(gamma=2, from_logits=True),
    r"focal, $\gamma = 5$": Focal(gamma=5, from_logits=True),
    "hinge": Hinge(),
    "squared hinge": Hinge(squared=True),
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, loss in losses.items():
    ax[0].plot(z, elementwise(loss.loss, z), label=name)
    if not isinstance(loss, Hinge):
        ax[1].plot(z, -elementwise(loss.gradient, z), label=name)
ax[0].set_ylim(0, 5)
ax[0].set_ylabel("loss")
ax[1].set_ylabel("negative gradient with respect to the logit")
for a in ax:
    a.set_xlabel("logit $z$ of a positive sample")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()

# %%
# Imbalanced binary classification
# --------------------------------
#
# Only 2 % of the samples are positive. The loss of a network trained with the binary cross entropy is dominated
# by the many easy negative samples. The focal loss down-weights the easy samples, so that the training focuses on
# the hard samples near the decision boundary, and alpha = 0.75 gives the positive class a larger weight than the
# negative class. The left figure shows the predicted probabilities of the positive test samples, and the right
# figure the ROC curves, which measure how well the predicted probabilities rank the positive samples above the
# negative ones independent of the chosen threshold.

X, y = make_classification(n_samples=6000, n_features=10, n_informative=4, weights=[0.98, 0.02], class_sep=1.0, flip_y=0, random_state=2)
X = StandardScaler().fit_transform(torch.tensor(X, dtype=torch.float32))
y = torch.tensor(y, dtype=torch.float32)
X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y, train_split=0.6, validation_split=0.2)


def binary_network(loss):
    model = Model(X.shape[1])
    model.add(Dense(32, activation=ReLU()))
    model.add(Dense(32, activation=ReLU()))
    model.add(Dense(0))  # logits, no activation
    model.compile(optimiser=ADAM(learning_rate=0.003), loss=loss, metrics=["loss", "val_loss"])
    return model


binary_losses = {
    "cross entropy": BCE(from_logits=True),
    r"focal, $\gamma = 2$, $\alpha = 0.75$": Focal(gamma=2, alpha=0.75, from_logits=True),
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, loss in binary_losses.items():
    torch.manual_seed(0)
    model = binary_network(loss)
    model.fit(X_train, y_train, val_data=(X_val, y_val), epochs=60, batch_size=64)
    # the model returns logits, which the loss turns to probabilities
    probabilities = model.loss.transform_prediction(model.predict(X_test))
    predictions = (probabilities > 0.5).to(torch.int32)
    fpr, tpr = roc_curve(probabilities, y_test, torch.linspace(0, 1, 500))
    print(f"{name}: precision {precision(predictions, y_test):.3f}, recall {recall(predictions, y_test):.3f}, ROC AUC {auc(fpr, tpr):.3f}")
    ax[0].hist(probabilities[y_test == 1].numpy(), bins=20, range=(0, 1), alpha=0.5, label=name)
    ax[1].plot(fpr, tpr, label=f"{name}, AUC = {auc(fpr, tpr):.3f}")
ax[0].set_xlabel("predicted probability of a positive test sample")
ax[0].set_ylabel("count")
ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
ax[1].set_xlabel("false positive rate")
ax[1].set_ylabel("true positive rate")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()

# %%
# The focal loss stops pushing the well classified samples towards the probabilities 0 and 1, so its predictions
# are less extreme than those of the cross entropy. Here it finds a few more of the positive samples at the
# threshold 0.5 (a higher recall) at the cost of precision, while the ROC curves of the two networks are almost
# identical. How much the focal loss helps depends on the dataset, so it is worth trying both.
#
# Soft labels and the multi-class hinge loss
# ------------------------------------------
#
# The labels of the cross entropy do not have to be one-hot vectors. In label smoothing, the one-hot labels are
# replaced by :math:`(1 - \varepsilon)y + \varepsilon / K`, where :math:`K` is the number of classes. The network is
# then not rewarded for pushing its predictions all the way to 0 or 1, which makes it less overconfident. The
# multi-class hinge loss trains the network like a support vector machine: the score of the correct class should be
# at least 1 larger than the score of every other class.

digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32) / 16
y = OneHotEncoder().fit_encode(torch.tensor(digits.target))
X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y, train_split=0.6, validation_split=0.2)
n_classes = y.shape[1]
epsilon = 0.2
y_train_smooth = (1 - epsilon) * y_train + epsilon / n_classes

settings = {
    "one-hot labels": (CCE(from_logits=True), y_train),
    rf"smoothed labels, $\varepsilon = {epsilon}$": (CCE(from_logits=True), y_train_smooth),
    "multi-class hinge": (Hinge(categorical=True), y_train),
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, (loss, labels) in settings.items():
    torch.manual_seed(0)
    model = Model(X.shape[1])
    model.add(Dense(64, activation=ReLU()))
    model.add(Dense(n_classes))  # logits or scores, no activation
    model.compile(optimiser=ADAM(learning_rate=0.003), loss=loss, metrics=["loss", "val_accuracy"])
    history = model.fit(X_train, labels, val_data=(X_val, y_val), epochs=40, batch_size=64)
    scores = model.predict(X_test)
    print(f"{name}: test accuracy {accuracy(scores, y_test):.3f}")
    ax[0].plot(history["val_accuracy"], label=name)
    if isinstance(loss, CCE):
        confidence = model.loss.transform_prediction(scores).max(dim=1).values
        ax[1].hist(confidence.numpy(), bins=20, range=(0, 1), alpha=0.5, label=name)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("validation accuracy")
ax[1].set_xlabel("largest predicted probability of a test sample")
ax[1].set_ylabel("count")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# The three networks reach a similar accuracy. With the smoothed labels, the largest predicted probability stays
# clearly below 1, while the network trained with the one-hot labels predicts nearly every sample with almost full
# confidence. The hinge loss gives scores instead of probabilities, so it is left out of the right figure.
