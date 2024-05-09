import torch
from Layers.RNN import RNN
from Model import Model
from Layers.Activations.ReLU import ReLU
from Layers.Activations.Tanh import Tanh
import matplotlib.pyplot as plt


size = 200
sequence_length = 25
x_data = torch.linspace(0, 10 * torch.pi, size)
y_data = torch.sin(x_data)
x = []
y = []
# for start in range(size - sequence_length):
#     x.append(data[start:start + sequence_length])
#     y.append(data[start + sequence_length])
# x = torch.stack(x).reshape(len(x), sequence_length, 1)
# y = torch.stack(y).reshape(len(y), 1)
# x = torch.stack(x).reshape(len(x), sequence_length, 1)
# y = torch.stack(y).reshape(len(y), 1)
for start in range(size - sequence_length):
    x.append(x_data[start:start + sequence_length])
    y.append(y_data[start:start + sequence_length])
x = torch.stack(x).reshape(len(x), sequence_length, 1)
y = torch.stack(y).reshape(len(y), sequence_length, 1)
print(x.shape, y.shape)

model = Model(1)
model.add(RNN(1, 100))
model.compile()
model.summary()

model.fit(x, y, epochs=100, batch_size=5)

x = torch.linspace(0, 10 * torch.pi, sequence_length)
y = torch.sin(x)

plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), model.predict(x.reshape(1, len(x), 1)).flatten().numpy())
plt.show()
