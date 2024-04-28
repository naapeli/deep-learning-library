import torch
from Layers.Conv2D import Conv2D


data = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
data = torch.stack([data, data, data])
data = data.reshape((3, 1, 5, 4))
target = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=torch.float32)
target = torch.stack([target, target, target]) + 1

layer = Conv2D(2, 2, input_shape=(1, 5, 4))
layer.initialise_layer()
output1 = layer.forward(data)
print(output1)
print(output1.shape)
print(target.shape)
print(layer.backward(target))

data = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32, requires_grad=True)
data = torch.stack([data, data, data])
data = data.reshape((3, 1, 5, 4))
data.retain_grad()

output = layer.forward(data)
output.backward(target)
print(data.grad)
