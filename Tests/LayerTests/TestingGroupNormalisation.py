import torch
from time import perf_counter

from src.DLL.DeepLearning.Layers.Regularisation import GroupNorm, InstanceNorm, LayerNorm


data = torch.rand((11, 10, 12, 15, 26), dtype=torch.float32, requires_grad=True)

layer = GroupNorm(num_groups=5)
# layer = InstanceNorm()
# layer = LayerNorm()
layer.initialise_layer(input_shape=(10, 12, 15, 26), data_type=torch.float32, device=torch.device("cpu"))
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
