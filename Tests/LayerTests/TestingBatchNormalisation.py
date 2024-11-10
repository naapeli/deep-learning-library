import torch
from time import perf_counter

from src.DLL.DeepLearning.Layers.Regularisation import BatchNorm


data = torch.rand((10, 11, 12), dtype=torch.float32, requires_grad=True)

layer = BatchNorm()
layer.initialise_layer(input_shape=(11, 12), data_type=torch.float32, device=torch.device("cpu"))
output = layer.forward(data, training=True)
target = torch.rand(size=output.size())
start = perf_counter()
output.backward(target)
print(f"Torch.auto_grad time: {perf_counter() - start}")
start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")

print(f"Backward pass # of wrong numbers: {torch.sum(data.grad != data_grad).item()}")
print(f"All elements close: {torch.allclose(data.grad, data_grad, rtol=0.0001)}")
print(f"Norm of the difference between gradients: {torch.linalg.norm(data.grad - data_grad)}")
