import torch
from time import perf_counter

from src.DLL.DeepLearning.Layers.Regularisation import Dropout


data = torch.rand((47, 46, 45), dtype=torch.float32, requires_grad=True)

layer = Dropout(0.5)
layer.initialise_layer(input_shape=(47, 46, 45), data_type=torch.float32, device=torch.device("cpu"))
output = layer.forward(data, training=True)
target = torch.rand_like(output)

start = perf_counter()
output.backward(target)
print(f"Torch.auto_grad time: {perf_counter() - start}")

start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")

# print(data.grad[0, 0])
# print(data_grad[0, 0])
print(f"Number of wrong numbers: {torch.sum(data.grad != data_grad).item()}")
print(f"All elements close: {torch.allclose(data.grad, data_grad)}")
print(f"Norm of the difference between gradients: {torch.linalg.norm(data.grad - data_grad)}")
