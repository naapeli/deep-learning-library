import torch
from time import perf_counter
from DeepLearning.Layers.Regularisation.BatchNormalisation import BatchNorm


data = torch.rand((10, 11, 12), dtype=torch.float32, requires_grad=True)

layer = BatchNorm(output_shape=(10, 11, 12))
output = layer.forward(data, training=True)
target = torch.rand(size=output.size())
start = perf_counter()
output.backward(target)
print(f"Torch.auto_grad time: {perf_counter() - start}")
start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")

# Working if biased variance is used
print(f"Backward pass # of wrong numbers: {torch.sum(data.grad != data_grad).item()}")
print(f"Norm of the difference between gradients: {torch.linalg.norm(data.grad - data_grad)}")

# print(output.mean(0))
# print(output.var(0))
