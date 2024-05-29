import torch
from time import perf_counter
from DeepLearning.Layers.Regularisation.GroupNormalisation import GroupNorm1d


data = torch.rand((11, 10, 12, 15, 26), dtype=torch.float32, requires_grad=True)

layer = GroupNorm1d(output_shape=(11, 10, 12, 15, 26), num_groups=5)
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
