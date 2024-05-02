import torch
from Layers.Conv2D import Conv2D
from Layers.Activations.ReLU import ReLU
from Layers.MaxPooling2D import MaxPooling2D
from torch.nn import MaxPool2d


data = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
data = torch.stack([data, data, data])
data = data.reshape((3, 1, 5, 4))
target = torch.tensor([[[0], [0]], [[0], [0]]], dtype=torch.float32)
target = torch.stack([target, target, target]) + 1

layer = Conv2D(2, 2, input_shape=(1, 5, 4), activation=ReLU())
layer2 = MaxPooling2D(2, input_shape=layer.output_shape)
output1 = layer.forward(data)
output2 = layer2.forward(output1)
print(output1)
print(output1.shape)
print(target.shape)
print(layer2.backward(target))

data = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32, requires_grad=True)
data = torch.stack([data, data, data])
data = data.reshape((3, 1, 5, 4))

output1 = layer.forward(data)
output1.retain_grad()
output2 = layer2.forward(output1)
output2.backward(target)
print(output1.grad)
