import torch
from Layers.RNN import RNN


data = torch.ones(size=(3, 1), dtype=torch.float32, requires_grad=True)
data = torch.stack([(i + 1) * data for i in range(2)])
data.retain_grad()
# print(data.shape)
# print(data)

layer = RNN(1, 5, input_shape=1)
layer.initialise_layer()
layer.bo.requires_grad = True
layer.bo.retain_grad()
result = layer.forward(data)
# print(result)
gradient = torch.ones_like(result)
result.backward(gradient)
print(data.grad)

print(layer.backward(gradient))
