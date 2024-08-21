import torch
import matplotlib.pyplot as plt

from MachineLearning.GaussianProcesses.GaussianProcessRegressor import GaussianProcessRegressor
from MachineLearning.GaussianProcesses.Kernels import SquaredExponentialCovariance, LinearCovariance, WhiteGaussianCovariance, PeriodicCovariance, RationalQuadraticCovariance
from DeepLearning.Optimisers.SGD import sgd
from DeepLearning.Optimisers.ADAM import Adam


X = torch.linspace(0, 1, 20, dtype=torch.float64).unsqueeze(1)
Y = 0.1 * torch.sin(20 * X) + X ** 2

model = GaussianProcessRegressor(LinearCovariance() ** 2 + PeriodicCovariance(1, 2, period=0.31) + WhiteGaussianCovariance(sigma=0.1) * LinearCovariance(), noise=0.1)
model.fit(X, Y)
print(model.log_marginal_likelihood())
model.train_kernel(epochs=100, optimiser=Adam())

x_test = torch.linspace(0, 2, 100, dtype=torch.float64).unsqueeze(1)
mean, covariance = model.predict(x_test)
mean = mean.squeeze()
std = torch.sqrt(torch.diag(covariance))
plt.plot(X, Y, ".")
plt.plot(x_test, mean, color="blue")
plt.plot(x_test, mean - 1.96 * std, alpha=0.5, color="red")
plt.plot(x_test, mean + 1.96 * std, alpha=0.5, color="red")
plt.show()

# draw random samples from the distribution
plt.plot(X, Y, ".")
distribution = torch.distributions.MultivariateNormal(mean, covariance)
for _ in range(5):
    y = distribution.sample()
    plt.plot(x_test, y)
plt.show()
