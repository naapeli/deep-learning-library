import torch

from src.DLL.DeepLearning.Layers.LSTM import LSTM

torch.manual_seed(0)
data = torch.ones(size=(5, 1), dtype=torch.float64, requires_grad=True)
data = torch.stack([(i + 1) * data for i in range(1)])
data.retain_grad()
# print(data.shape)
# print(data)

layer = LSTM((None, 1), 5, input_shape=(None, None, 1), data_type=torch.float64)
layer.initialise_layer()
result = layer.forward(data)
# print(result)
gradient = torch.rand(size=result.size(), dtype=torch.float64)
result.backward(gradient)
print(data.grad)

gradient = layer.backward(gradient)
print(gradient)

