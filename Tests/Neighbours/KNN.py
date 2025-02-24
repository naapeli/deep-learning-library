import torch
import matplotlib.pyplot as plt
from sklearn import datasets

from src.DLL.Data.Metrics import accuracy
from src.DLL.Data.Preprocessing import data_split
from src.DLL.MachineLearning.SupervisedLearning.Neighbors import KNNClassifier, KNNRegressor


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


X = torch.linspace(0, 1, 100)
y = X ** 2 + torch.randn_like(X) * 0.05
X = X.unsqueeze(1)
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model = KNNRegressor(k=5, metric="manhattan", weight="gaussian")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

plt.scatter(X_test, predictions, label="predictions")
plt.scatter(X_test, y_test, label="true")
plt.scatter(X_train, y_train, label="train")
plt.legend()
plt.show()
