import torch
from sklearn import datasets

from src.DLL.MachineLearning.LinearModels.LogisticRegression import LogisticRegression
from src.DLL.Data.Preprocessing import MinMaxScaler, OneHotEncoder, data_split
from src.DLL.Data.Metrics import accuracy


iris = datasets.load_iris()

scaler = MinMaxScaler()
encoder = OneHotEncoder()
x = torch.tensor(iris.data, dtype=torch.float32)
scaler.fit(x)
x = scaler.transform(x)
y = torch.tensor(iris.target, dtype=torch.float32)
encoder.fit(y)
y = encoder.one_hot_encode(y)
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.7, validation_split=0.0)

model = LogisticRegression(learning_rate=0.001)
model.fit(x_train, y_train, epochs=2000, metrics=["loss", "accuracy"], callback_frequency=10)
y_pred = model.predict(x_test)
print(accuracy(y_pred, y_test))