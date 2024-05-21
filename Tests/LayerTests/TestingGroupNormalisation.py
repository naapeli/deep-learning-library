import torch
from DeepLearning.Layers.Regularisation.GroupNormalisation import GroupNorm1d

layer = GroupNorm1d(output_shape=(1, 4), num_groups=1)
data = torch.tensor([[1, 2, 3, 5], [5, 6, 7, 9], [10, 11, 13, 15]], dtype=torch.float32)
output = layer.forward(data)
loss = torch.tensor([[-1.1381, -0.1953, -1.1381, -0.1953], [-1.8047, -0.8619, -1.8047, -0.8619], [-2.4714, -1.5286, -2.4714, -1.5286]])
print(layer.backward(loss))

layer = GroupNorm1d(output_shape=(1, 4), num_groups=1)
data = torch.tensor([[1, 2, 3, 5], [5, 6, 7, 9], [10, 11, 13, 15]], dtype=torch.float32, requires_grad=True)
loss = torch.tensor([[-1.1381, -0.1953, -1.1381, -0.1953], [-1.8047, -0.8619, -1.8047, -0.8619], [-2.4714, -1.5286, -2.4714, -1.5286]])
output = layer.forward(data)
output.backward(loss)
print(data.grad)

# forward pass working (if biased variance is used for the calculation)
print(torch.nn.GroupNorm(2, 4)(torch.tensor([[1, 2, 3, 5], [5, 6, 7, 9]], dtype=torch.float32)))
