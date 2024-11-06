import torch
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers import Dense
from src.DLL.DeepLearning.Layers.Regularisation import BatchNorm, GroupNorm, InstanceNorm, LayerNorm, Dropout
from src.DLL.DeepLearning.Layers.Activations import Tanh, Sigmoid, ReLU
from src.DLL.DeepLearning.Losses import mse
from src.DLL.DeepLearning.Optimisers import sgd
from src.DLL.Data.Preprocessing import data_split


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model(2, data_type=torch.float32)
model.add(Dense(6, initialiser="Xavier_uniform", normalisation=BatchNorm(), activation=ReLU()))
model.add(Dropout(6, p=0.1))
model.add(Dense(6, initialiser="He_norm"))
model.add(BatchNorm(6))
model.add(Sigmoid(6))
model.add(Dense(0, initialiser="Xavier_norm"))
model.compile(optimiser=sgd(learning_rate=0.1), loss=mse(), metrics=["loss", "val_loss"])
model.summary()
n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float32, device=device), torch.linspace(-1, 1, n, dtype=torch.float32, device=device), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=20, batch_size=64, verbose=True)
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
