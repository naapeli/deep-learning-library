"""
Regression with neural networks
==================================

This script implements a model to predict values on a simple quadratic surface. It also 
showcases some regularisation methods like Dropout and BatchNorm.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense, Identity, Add
from DLL.DeepLearning.Layers.Regularisation import BatchNorm, Dropout
from DLL.DeepLearning.Layers.Activations import Tanh, Sigmoid, ReLU
from DLL.DeepLearning.Losses import MSE
from DLL.DeepLearning.Optimisers import SGD, RMSPROP
from DLL.DeepLearning.Initialisers import Xavier_Normal, Xavier_Uniform, Kaiming_Normal, Kaiming_Uniform
from DLL.Data.Preprocessing import data_split


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model(2, data_type=torch.float32, device=device)
model.add(Dense(6, initialiser=Xavier_Uniform(), normalisation=BatchNorm(), activation=ReLU()))
model.add(Add(Dense(6, activation=ReLU()), Identity(), normalisation=BatchNorm()))
model.add(Dropout(p=0.1))
model.add(Dense(6, initialiser=Kaiming_Normal()))
model.add(BatchNorm())
model.add(Sigmoid())
model.add(Dense(0, initialiser=Xavier_Normal()))
# model.compile(optimiser=SGD(learning_rate=0.1), loss=MSE(), metrics=["loss", "val_loss", "median_absolute"])
model.compile(optimiser=RMSPROP(learning_rate=0.01), loss=MSE(), metrics=["loss", "val_loss", "median_absolute"])
model.summary()

n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float32, device=device), torch.linspace(-1, 1, n, dtype=torch.float32, device=device), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size(), device=device) - 5
x_train, y_train, x_val, y_val, x_test, y_test = data_split(x, y, train_split=0.6, validation_split=0.2)

errors = model.fit(x_train, y_train, val_data=(x_val, y_val), epochs=50, batch_size=64, verbose=True)

plt.figure(figsize=(8, 8))
plt.semilogy(errors["loss"], label="loss")
plt.semilogy(errors["val_loss"], label="val_loss")
plt.semilogy(errors["median_absolute"], label="median absolute error")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Metric")
z = model.predict(x_test).cpu().numpy()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(x_test[:, 0].cpu().numpy(), x_test[:, 1].cpu().numpy(), z, color="blue")
surf = ax.scatter(x_test[:, 0].cpu().numpy(), x_test[:, 1].cpu().numpy(), y_test.cpu().numpy(), color="red")
plt.show()
