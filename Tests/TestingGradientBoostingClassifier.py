import torch
from sklearn import datasets
from sklearn.metrics import auc as sk_auc
import matplotlib.pyplot as plt
import numpy as np

from src.DLL.MachineLearning.SupervisedLearning.Trees import GradientBoostingClassifier
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy, roc_curve, auc


n_classes = 2
np.random.seed(0)
X, y = datasets.make_blobs(n_features=2, n_samples=1000, centers=n_classes, cluster_std=2)
plt.scatter(X[:, 0], X[:, 1], c=y)

x_train, y_train, _, _, x_test, y_test = data_split(torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y), train_split=0.7, validation_split=0.0)

model = GradientBoostingClassifier(n_trees=3, learning_rate=0.5, loss="log_loss")
model.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)
y_pred = model.predict(x_test)
print(y_pred[:20])
print(y_test[:20])
print(y_pred_proba[:20])
print(accuracy(y_pred, y_test))

if n_classes == 2:
    thresholds = torch.linspace(0, 1, 100)
    fpr, tpr = roc_curve(y_pred_proba, y_test, thresholds)
    plt.figure()
    plt.title(f"ROC curve with auc = {auc(fpr, tpr)}")
    print(sk_auc(fpr, tpr))
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

plt.figure()
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=((model.predict(x_test) != y_test) + 0.2) / 1.2)

plt.show()
