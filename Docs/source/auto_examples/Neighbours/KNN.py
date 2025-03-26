"""
K-Nearest Neighbors (KNN) Classification and Regression
============================================================

This script demonstrates the use of K-Nearest Neighbors (KNN) for both classification and regression 
tasks using the `KNNClassifier` and `KNNRegressor` models. It also showcases model serialization 
with `save_model`.
"""
import torch
import matplotlib.pyplot as plt
from sklearn import datasets

from DLL.Data.Metrics import accuracy
from DLL.Data.Preprocessing import data_split
from DLL.MachineLearning.SupervisedLearning.Neighbors import KNNClassifier, KNNRegressor
from DLL.DeepLearning.Model import save_model, load_model


torch.manual_seed(0)

iris = datasets.load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.float32)
# X = X[y != 2]
# y = y[y != 2]
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model = KNNClassifier(k=50, metric="manhattan")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.predict_proba(X_test))
print(accuracy(predictions, y_test))
# save_model(model, filepath="./Tests/Neighbours/classifier.pkl")


X = torch.linspace(0, 1, 100)
y = X ** 2 + torch.randn_like(X) * 0.05
X = X.unsqueeze(1)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model = KNNRegressor(k=5, metric="manhattan", weight="gaussian")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# save_model(model, filepath="./Tests/Neighbours/regressor.pkl")

plt.scatter(X_test, predictions, label="predictions")
plt.scatter(X_test, y_test, label="true")
plt.scatter(X_train, y_train, label="train")
plt.legend()
plt.show()
