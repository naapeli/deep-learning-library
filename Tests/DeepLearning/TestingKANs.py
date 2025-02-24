import torch
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Layers._DenseKAN import _get_basis_functions, _NeuronKAN
from src.DLL.DeepLearning.Layers import DenseKAN
from src.DLL.DeepLearning.Layers.Activations import Tanh
from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Optimisers import ADAM
from src.DLL.DeepLearning.Losses import MSE
from src.DLL.DeepLearning.Initialisers import Xavier_Normal
from src.DLL.Data.Preprocessing import data_split

# n_fun = 10
# basis_funcs, basis_func_derivatives = _get_basis_functions(n_fun, degree=5, bounds=(-1, 1))
# x = torch.linspace(-1, 1, 100).unsqueeze(1)

# plt.figure(figsize=(10, 5))
# for i in range(n_fun):
#     basis_values = basis_funcs[i](x)
#     derivative_values = basis_func_derivatives[i](x)

#     label = f"B-spline basis (i={i}, n_fun={n_fun})" if i > 0 else "Silu"
#     plt.plot(x.squeeze(1), basis_values, label=label)
#     label = f"Derivative (i={i}, n_fun={n_fun})" if i > 0 else "Silu derivative"
#     # plt.plot(x, derivative_values, label=label)

# plt.title("B-spline Basis Functions and Derivatives")
# plt.xlabel("x")
# plt.ylabel("Value")
# plt.legend()
# plt.grid()
# plt.show()


# x = torch.rand(size=(100, 20))
# x.requires_grad = True

# neuron = _NeuronKAN(20, 11, (-1, 1), Xavier_Normal(), torch.float32, torch.device("cpu"))
# result = neuron.forward(x)
# loss = torch.randn_like(result)
# result.backward(loss)
# print(x.grad)
# gradient = neuron.backward(loss)
# print(gradient)
# print(x.grad - gradient)

# layer = DenseKAN(11)
# layer.initialise_layer((20,), torch.float32, torch.device("cpu"))
# result = layer.forward(x)
# loss = torch.randn_like(result)
# result.backward(loss)
# print(x.grad)
# gradient = layer.backward(loss)
# print(gradient)
# print(x.grad - gradient)

# X = torch.linspace(-1, 1, 100).unsqueeze(1)
# y = torch.sin(4 * X).squeeze()
# X = torch.linspace(-1, 1, 50).unsqueeze(1)
# y = (0.5 * torch.sin(4 * X) * torch.exp(-X - 1) + 0.5).squeeze()
n = 30
X, Y = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
X, y, _, _, X_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)


model = Model(2)
# model.add(DenseKAN(1, activation=Tanh(), initialiser=Xavier_Normal()))
model.add(DenseKAN(2, activation=Tanh(), initialiser=Xavier_Normal()))
# model.add(DenseKAN(2, activation=Tanh(), initialiser=Xavier_Normal()))
model.add(DenseKAN(0, initialiser=Xavier_Normal()))
model.compile(ADAM(0.01), MSE())
model.summary()

history = model.fit(X, y, epochs=300, verbose=True)

# X_test = 2 * torch.rand_like(X) - 1
# y_test = torch.sin(4 * X_test).squeeze()
# X_test = 2 * torch.rand_like(X) - 1
# y_test = (0.5 * torch.sin(4 * X_test) * torch.exp(-X_test - 1) + 0.5).squeeze()

# plt.figure()
# plt.scatter(X_test, y_test, label="True test points")
# plt.scatter(X_test, model.predict(X_test), label="Predictions")
# plt.scatter(X, y, label="True train points")
# plt.scatter(X, model.predict(X), label="Predicted train points")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="True test points")
ax.scatter(X_test[:, 0], X_test[:, 1], model.predict(X_test), label="Predictions")
plt.legend()


plt.figure()
plt.plot(history["loss"])

plt.show()
