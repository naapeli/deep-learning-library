from Model import Model
from Layers.Dense import Dense
from Layers.Activations.Tanh import Tanh
from Layers.Activations.ReLU import ReLU
from Losses.MSE import mse
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32, device=device)

model = Model(2)
model.add(Dense(3, activation=Tanh()))
model.add(Dense(1, activation=Tanh()))
model.compile(optimiser=None, loss=mse())
model.summary()
errors = model.fit(x.reshape(1, 4, 2), y.reshape(1, 4), epochs=1000, loss_step=100)
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")

Xv, Yv = torch.meshgrid(torch.linspace(0, 1, 50, dtype=torch.float32, device=device), torch.linspace(0, 1, 50, dtype=torch.float32, device=device), indexing="xy")
z = model.predict(torch.stack([Xv.flatten(), Yv.flatten()]).T)
z = z.reshape(Xv.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(Xv, Yv, z)
fig.colorbar(surf)
plt.show()
