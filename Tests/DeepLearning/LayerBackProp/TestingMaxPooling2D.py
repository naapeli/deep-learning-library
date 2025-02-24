import torch
from torch.nn import MaxPool2d
from time import perf_counter

from src.DLL.DeepLearning.Layers import MaxPooling2D


data = torch.rand((50, 50, 50, 50), dtype=torch.float32, requires_grad=True)
data = torch.where(data == data[0, 0, 0, 0], torch.tensor(2), data)
data = torch.where(data == data[0, 0, 0, 1], torch.tensor(2), data)
data.retain_grad()
#print(data)

layer = MaxPooling2D(3)
layer.initialise_layer((50, 50, 50), data_type=torch.float32, device=torch.device("cpu"))
output = layer.forward(data)
target = torch.ones_like(output)
start = perf_counter()
output.backward(target)
print(f"Torch.auto_grad time: {perf_counter() - start}")
#print(data.grad)

start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")
#print(data_grad)

#torch_model = MaxPool2d(3)
#print(output)
#print(torch_model.forward(data.clone()))
print(f"Number of wrong numbers: {torch.sum(data.grad != data_grad).item()}")
