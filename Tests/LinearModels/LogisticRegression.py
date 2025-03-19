import torch
from sklearn import datasets
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.SupervisedLearning.LinearModels import LogisticRegression
from src.DLL.Data.Preprocessing import MinMaxScaler, CategoricalEncoder, data_split
from src.DLL.Data.Metrics import accuracy


iris = datasets.load_iris()

binary = False
if binary:
    x = torch.tensor(iris.data[iris.target != 2], dtype=torch.float32)
    y = torch.tensor(iris.target[iris.target != 2], dtype=torch.float32)
else:
    x = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.float32)
scaler = MinMaxScaler()
encoder = CategoricalEncoder()
x = scaler.fit_transform(x)
y = encoder.fit_encode(y)
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.7, validation_split=0.0)

model = LogisticRegression(learning_rate=0.001)
history = model.fit(x_train, y_train, epochs=5000, metrics=["loss", "accuracy"], callback_frequency=100, verbose=True)
y_pred = model.predict(x_test)
print(y_pred, model.predict_proba(x_test))
print(y_pred, y_test)
print(accuracy(y_pred, y_test))

plt.plot(history["accuracy"])
plt.xlabel("epoch / 10")
plt.ylabel("accuracy")

plt.figure()
plt.semilogy(history["loss"])
plt.xlabel("epoch / 10")
plt.ylabel("loss")
plt.show()
