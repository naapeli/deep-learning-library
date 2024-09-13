import torch
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers.Dense import Dense
from src.DLL.DeepLearning.Layers.Regularisation.Dropout import Dropout
from src.DLL.DeepLearning.Layers.Regularisation.BatchNormalisation import BatchNorm
from src.DLL.DeepLearning.Layers.Regularisation.GroupNormalisation import GroupNorm
from src.DLL.DeepLearning.Layers.Regularisation.InstanceNormalisation import InstanceNorm
from src.DLL.DeepLearning.Layers.Regularisation.LayerNormalisation import LayerNorm
from src.DLL.DeepLearning.Layers.Activations.Tanh import Tanh
from src.DLL.DeepLearning.Layers.Activations.ReLU import ReLU
from src.DLL.DeepLearning.Layers.Activations.Sigmoid import Sigmoid
from src.DLL.DeepLearning.Losses.MSE import mse
from src.DLL.DeepLearning.Optimisers.SGD import sgd
from src.DLL.Data.Preprocessing import data_split


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model((None, 2), data_type=torch.float32)
model.add(Dense((None, 6), normalisation=BatchNorm(), activation=ReLU()))
model.add(Dropout((None, 6), p=0.1))
model.add(Dense((None, 6)))
model.add(BatchNorm((None, 6)))
model.add(Sigmoid((None, 6)))
model.add(Dense((None, 1)))
model.compile(optimiser=sgd(learning_rate=0.1), loss=mse(), metrics=["loss", "val_loss"])
model.summary()
n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float32, device=device), torch.linspace(-1, 1, n, dtype=torch.float32, device=device), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=100, batch_size=64)
plt.plot(errors["loss"], label="loss")
plt.plot(errors["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
z = model.predict(x_test)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(x_test[:, 0], x_test[:, 1], z, color="blue")
surf = ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color="red")
plt.show()
