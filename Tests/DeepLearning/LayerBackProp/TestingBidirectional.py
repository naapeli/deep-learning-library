import torch
import torch.nn as nn

from src.DLL.DeepLearning.Layers import RNN, LSTM, Bidirectional


data = torch.rand(size=(3, 1), dtype=torch.float32, requires_grad=True)
data = torch.stack([(i + 1) * data for i in range(2)])
data.retain_grad()
# print(data.shape)
# print(data)

layer = Bidirectional(RNN(1, 5, return_last=False))
layer.initialise_layer((1,), data_type=torch.float32, device=torch.device("cpu"))
result = layer.forward(data)
# print(result)
gradient = torch.ones_like(result)
result.backward(gradient)
print(data.grad)
print(layer.backward(gradient))
