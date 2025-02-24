import torch
from time import perf_counter

from src.DLL.DeepLearning.Layers import Conv2D


data = torch.rand((46, 47, 48, 49), dtype=torch.float32, requires_grad=True)

layer = Conv2D(3, 1)
layer.initialise_layer(input_shape=(47, 48, 49), data_type=torch.float32, device=torch.device("cpu"))
output = layer.forward(data)
target = torch.ones_like(output)

start = perf_counter()
output.backward(target)
print(f"Torch.auto_grad time: {perf_counter() - start}")

start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")

print(f"Number of wrong numbers: {torch.sum(data.grad != data_grad).item()}")
print(f"All elements close: {torch.allclose(data.grad, data_grad)}")
print(f"Norm of the difference between gradients: {torch.linalg.norm(data.grad - data_grad)}")
