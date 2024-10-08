import torch
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Layers.RNN import RNN
from src.DLL.DeepLearning.Layers.LSTM import LSTM
from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers.Activations.Tanh import Tanh
from src.DLL.DeepLearning.Optimisers.ADAM import Adam
from src.DLL.DeepLearning.Losses.MSE import mse


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
model.add(RNN((None, 1), 10, activation=Tanh()))
model.compile(optimiser=Adam(), loss=mse(), metrics=["loss", "val_loss"])
model.summary()

model.fit(x, y, epochs=20, batch_size=1, verbose=True)

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
