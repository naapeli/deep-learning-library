"""
Recurrent networks for time series analysis
============================================

This script implements a model to predict values of a simple sine function. It uses recurrent layers 
to handle the sequential nature of the sine function.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Layers import RNN, LSTM
from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import MSE


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

size = 300
sequence_length = 20
data = torch.sin(torch.arange(size, dtype=torch.float32))
x = []
y = []
for start in range(size - sequence_length):
    x.append(data[start:start + sequence_length])
    y.append(data[start + sequence_length])
x = torch.stack(x).reshape(len(x), sequence_length, 1).to(device=device)
y = torch.stack(y).reshape(len(y), 1).to(device=device)
print(x.shape, y.shape)

model = Model((sequence_length, 1), device=device)
model.add(RNN((1,), 10, activation=Tanh()))
model.compile(optimiser=ADAM(), loss=MSE(), metrics=["loss", "val_loss"])
model.summary()

model.fit(x, y, epochs=20, batch_size=1, verbose=True)

data = torch.sin(torch.arange(size, dtype=torch.float32, device=device))
prediction = []
y_true = []
for start in range(30):
    prediction.append(model.predict(data[start:start + sequence_length].reshape(1, sequence_length, 1)))
    y_true.append(data[start + sequence_length])

plt.figure(figsize=(8, 8))
plt.plot(torch.stack(y_true).cpu().numpy(), label="Sin(x)")
plt.plot(torch.stack(prediction).cpu().flatten().numpy(), label="Prediction")
plt.legend()
plt.show()
