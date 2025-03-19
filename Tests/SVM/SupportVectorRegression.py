import torch
import matplotlib.pyplot as plt
from sklearn import svm

from src.DLL.MachineLearning.SupervisedLearning.SupportVectorMachines import SVR
from src.DLL.Data.Preprocessing import data_split
from src.DLL.MachineLearning.SupervisedLearning.Kernels import Linear


torch.manual_seed(10)

x = torch.linspace(-2, 2, 20)
y = torch.linspace(-2, 2, 20)
XX, YY = torch.meshgrid(x, y, indexing="xy")
X = XX.flatten()
Y = YY.flatten()
X_input = torch.stack((X, Y), dim=1)
Z = 2 * X ** 2 - 5 * Y ** 2 + torch.normal(0, 1, size=X.size())
x_train, y_train, x_test, y_test, _, _ = data_split(X_input, Z, train_split=0.8, validation_split=0.2)

model = SVR(kernel=Linear() * Linear())
# model = SVR()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

correct = svm.SVR(kernel="rbf", C=1)
correct.fit(x_train, y_train)
y_pred_true = correct.predict(x_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_test.numpy(), label="True")
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_pred.numpy(), label="Prediction")
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_pred_true, label="sklearn")
ax.legend()
plt.show()
