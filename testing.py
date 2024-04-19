from Model import Model
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Layers.Activations.Tanh import Tanh
from Layers.Activations.ReLU import ReLU
from Losses.MSE import mse
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device, requires_grad=False)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32, device=device, requires_grad=False)

model = Model(2, data_type=torch.float32)
model.add(Dense(3, activation=Tanh()))
# model.add(Dropout(3, p=0.9))
model.add(Dense(1, activation=Tanh()))
model.compile(optimiser=None, loss=mse())
model.summary()
errors = model.fit(x, y, epochs=10000, loss_step=10, batch_size=2, shuffle_data=True, new_shuffle_per_epoch=True)
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
print(model.predict(x))

Xv, Yv = torch.meshgrid(torch.linspace(0, 1, 50, dtype=torch.float32, device=device, requires_grad=False), torch.linspace(0, 1, 50, dtype=torch.float32, device=device, requires_grad=False), indexing="xy")
z = model.predict(torch.stack([Xv.flatten(), Yv.flatten()]).T)
z = z.reshape(Xv.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(Xv, Yv, z)
plt.show()
