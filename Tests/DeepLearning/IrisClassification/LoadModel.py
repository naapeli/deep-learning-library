import torch
from sklearn import datasets
from src.DLL.Data.Preprocessing import OneHotEncoder, MinMaxScaler

from src.DLL.DeepLearning.Model import load_model
from src.DLL.Data.Metrics import accuracy



model = load_model(filepath="./Tests/DeepLearning/IrisClassification/model.pkl")


iris = datasets.load_iris()

encoder = OneHotEncoder()
scaler = MinMaxScaler()
x = torch.tensor(iris.data, dtype=torch.float32)
x = scaler.fit_transform(x)
y = encoder.fit_encode(torch.tensor(iris.target, dtype=torch.float32))

y_pred = model.predict(x)
print(accuracy(y_pred, y))
