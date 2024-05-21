import torch
from DeepLearning.Layers.RNN import RNN
from DeepLearning.Model import Model
from DeepLearning.Layers.Activations.Tanh import Tanh
import matplotlib.pyplot as plt


size = 300
sequence_length = 10
data = torch.sin(torch.arange(size, dtype=torch.float32))
x = []
y = []
for start in range(size - sequence_length):
    x.append(data[start:start + sequence_length])
    y.append(data[start + sequence_length])
x = torch.stack(x).reshape(len(x), sequence_length, 1)
y = torch.stack(y).reshape(len(y), 1)
print(x.shape, y.shape)

model = Model((None, None, 1))
model.add(RNN((None, 1), 100, activation=Tanh()))
model.compile()
model.summary()

model.fit(x, y, epochs=10, batch_size=1)

data = torch.sin(torch.arange(size, dtype=torch.float32))
prediction = []
y_true = []
for start in range(30):
    prediction.append(model.predict(data[start:start + sequence_length].reshape(1, sequence_length, 1)))
    y_true.append(data[start + sequence_length])

plt.plot(torch.stack(y_true).numpy(), label="Sin(x)")
plt.plot(torch.stack(prediction).flatten().numpy(), label="Prediction")
plt.legend()
plt.show()
