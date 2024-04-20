from Model import Model
from Layers.Dense import Dense
from Layers.Regularisation.Dropout import Dropout
from Layers.Regularisation.BatchNormalisation import BatchNorm1d
from Layers.Activations.Tanh import Tanh
from Layers.Activations.ReLU import ReLU
from Losses.MSE import mse
from Data.Processing import data_split

import torch
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model(2, data_type=torch.float32)
model.add(Dense(6, normalisation=None, activation=None))
model.add(BatchNorm1d(6))
model.add(ReLU(6))
# model.add(Dropout(6, p=0.9))
model.add(Dense(6))
model.add(BatchNorm1d(6))
model.add(Tanh(6))
model.add(Dense(1))
model.compile(optimiser=None, loss=mse())
model.summary()
n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float32, device=device, requires_grad=False), torch.linspace(-1, 1, n, dtype=torch.float32, device=device, requires_grad=False), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=50, batch_size=8, learning_rate=0.1, metrics=["loss", "val_loss"], shuffle_data=True, new_shuffle_per_epoch=True)
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
