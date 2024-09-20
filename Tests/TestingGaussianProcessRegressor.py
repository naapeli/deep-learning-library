import torch
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.GaussianProcesses.GaussianProcessRegressor import GaussianProcessRegressor
from src.DLL.MachineLearning.Kernels import SquaredExponentialCovariance, LinearCovariance, WhiteGaussianCovariance, PeriodicCovariance, RationalQuadraticCovariance
from src.DLL.DeepLearning.Optimisers.ADAM import Adam
from src.DLL.Data.Preprocessing import StandardScaler


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")  # cpu seems to perform faster for some reason
X = torch.linspace(0, 1, 20, dtype=torch.float64, device=device).unsqueeze(1)
Y = 0.1 * torch.sin(20 * X) + X ** 2
transformer = StandardScaler()
transformer.fit(Y)
Y = transformer.transform(Y)

model = GaussianProcessRegressor(LinearCovariance(sigma=0.2) ** 2 + PeriodicCovariance(1, 2, period=0.5), noise=0.1, device=device)
model.fit(X, Y)
print(model.log_marginal_likelihood())
model.train_kernel(epochs=200, optimiser=Adam())  # try to comment out this line of code to see how the covariance kernel learns the correct parameters

x_test = torch.linspace(0, 2, 100, dtype=torch.float64, device=device).unsqueeze(1)
mean, covariance = model.predict(x_test)
mean = mean.squeeze()
mean = transformer.inverse_transform(mean)
covariance = covariance * transformer.var ** 2
std = torch.sqrt(torch.diag(covariance))
plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
plt.plot(x_test.cpu(), mean.cpu(), color="blue")
plt.plot(x_test.cpu(), mean.cpu() - 1.96 * std.cpu(), alpha=0.5, color="red")
plt.plot(x_test.cpu(), mean.cpu() + 1.96 * std.cpu(), alpha=0.5, color="red")
plt.show()

# draw random samples from the distribution
plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
distribution = torch.distributions.MultivariateNormal(mean, covariance)
for _ in range(5):
    y = distribution.sample()
    plt.plot(x_test.cpu(), y.cpu())
plt.show()
