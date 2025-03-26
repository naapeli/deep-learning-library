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

model = load_model(filepath="./Tests/Neighbours/classifier.pkl")
predictions = model.predict(X)
print(model.predict_proba(X))
print(accuracy(predictions, y))


X = torch.linspace(0, 1, 100)
y = X ** 2 + torch.randn_like(X) * 0.05
X = X.unsqueeze(1)

model = load_model(filepath="./Tests/Neighbours/regressor.pkl")
predictions = model.predict(X)

plt.scatter(X, predictions, label="predictions")
plt.scatter(X, y, label="true")
plt.legend()
plt.show()
