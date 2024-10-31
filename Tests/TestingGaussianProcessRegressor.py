import torch
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.SupervisedLearning.GaussianProcesses.GaussianProcessRegressor import GaussianProcessRegressor
from src.DLL.MachineLearning.SupervisedLearning.Kernels import SquaredExponentialCovariance, LinearCovariance, WhiteGaussianCovariance, PeriodicCovariance, RationalQuadraticCovariance
from src.DLL.DeepLearning.Optimisers.ADAM import Adam
from src.DLL.Data.Preprocessing import StandardScaler


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")  # cpu seems to perform faster for some reason
X = torch.linspace(0, 1, 20, dtype=torch.float64, device=device).unsqueeze(1)
Y = 0.1 * torch.sin(20 * X) + X ** 2# + torch.randn_like(X) * 0.5
transformer = StandardScaler()
transformer.fit(Y)
Y = transformer.transform(Y).squeeze(dim=1)

train_kernel = True  # try to changing this line of code to see how the covariance kernel learns the correct parameters

model = GaussianProcessRegressor(LinearCovariance(sigma=0.2) ** 2 + PeriodicCovariance(1, 2, period=0.5), noise=0.1, device=device)
model.fit(X, Y)
print(model.log_marginal_likelihood())
if train_kernel:
    history = model.train_kernel(epochs=500, optimiser=Adam(), verbose=True)
    plt.plot(history["log marginal likelihood"])
    plt.xlabel("epoch")
    plt.ylabel("log marginal likelihood")
    plt.title("The change in log marginal likelihood during kernel training")
print([round(param.item(), 3) for param in model.covariance_function.parameters()])

x_test = torch.linspace(0, 2, 100, dtype=torch.float64, device=device).unsqueeze(1)
mean, covariance = model.predict(x_test)
mean = mean.squeeze()
mean = transformer.inverse_transform(mean)
covariance = covariance * transformer.var ** 2
std = torch.sqrt(torch.diag(covariance))
plt.figure()
plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
plt.plot(x_test.cpu(), mean.cpu(), color="blue", label="mean")
plt.fill_between(x_test.squeeze(dim=1).cpu(), mean.cpu() - 1.96 * std.cpu(), mean.cpu() + 1.96 * std.cpu(), alpha=0.1, color="blue", label=r"95% cofidence interval")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predictions with the GPR model")

# draw random samples from the distribution
blue_theme = [
    "#1f77b4",  # blue
    "#4a8cd3",  # lighter blue
    "#005cbf",  # dark blue
    "#7cb9e8",  # sky blue
    "#0073e6",  # vivid blue
    "#3b5998",  # muted blue
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=blue_theme)

plt.figure()
plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
distribution = torch.distributions.MultivariateNormal(mean, covariance)
for _ in range(5):
    y = distribution.sample()
    plt.plot(x_test.cpu(), y.cpu())
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random samples from the previous distribution")
plt.show()
