import torch
from sklearn import datasets
from sklearn.metrics import auc as sk_auc
import matplotlib.pyplot as plt
import numpy as np

from src.DLL.MachineLearning.SupervisedLearning.LinearModels import LogisticRegression
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy, roc_curve, auc


np.random.seed(0)
X, y = datasets.make_blobs(n_features=2, n_samples=1000, centers=2)
plt.scatter(X[:, 0], X[:, 1], c=y)

x_train, y_train, _, _, x_test, y_test = data_split(torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y), train_split=0.7, validation_split=0.0)

model = LogisticRegression(learning_rate=0.001)
model.fit(x_train, y_train, epochs=2000, metrics=["loss", "accuracy"], callback_frequency=10, verbose=True)
y_pred = model.predict_proba(x_test)
print(y_pred[:10], y_test[:10])
print(accuracy(y_pred, y_test))

thresholds = torch.linspace(0, 1, 100)
fpr, tpr = roc_curve(y_pred, y_test, thresholds)
plt.figure()
plt.title(f"ROC curve with auc = {auc(fpr, tpr)}")
print(sk_auc(fpr, tpr))
plt.plot([0, 1], [0, 1])
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()
