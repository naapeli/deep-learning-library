import torch
from DeepLearning.Layers.RNN import RNN


data = torch.ones(size=(3, 1), dtype=torch.float32, requires_grad=True)
data = torch.stack([(i + 1) * data for i in range(2)])
data.retain_grad()
# print(data.shape)
# print(data)

layer = RNN((None, None, 1), 5, input_shape=(None, 1))
layer.initialise_layer()
result = layer.forward(data)
# print(result)
gradient = torch.ones_like(result)
result.backward(gradient)
print(data.grad)

print(layer.backward(gradient))
