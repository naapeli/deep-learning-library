import torch
from time import perf_counter

from src.DLL.DeepLearning.Layers import MultiHeadAttention


data = torch.rand((47, 46, 45), dtype=torch.float32, requires_grad=True)

layer = MultiHeadAttention((46, 45), n_heads=5, use_mask=True, dropout=0.5)
layer.initialise_layer(input_shape=(46, 45), data_type=torch.float32, device=torch.device("cpu"))
output = layer.forward(data, training=True)
# print(output)
target = torch.rand_like(output)

start = perf_counter()
output.backward(target)
# Q, K, V  = input.chunk(3, dim=2)
# temp = torch.nn.MultiheadAttention(15, 3)
# print(temp(q, k, v)[0])
print(f"Torch.auto_grad time: {perf_counter() - start}")

start = perf_counter()
data_grad = layer.backward(target)
print(f"My implementation time: {perf_counter() - start}")

# print(data.grad[0, 0, 0])
# print(data_grad[0, 0, 0])
print(f"Number of wrong numbers: {torch.sum((~torch.isclose(data.grad, data_grad, rtol=0.001)).float()).int().item()}")
print(f"All elements close: {torch.allclose(data.grad, data_grad)}")
print(f"Norm of the difference between gradients: {torch.linalg.norm(data.grad - data_grad)}")
