import torch
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared, ConstantKernel, Matern as sk_matern

from src.DLL.MachineLearning.SupervisedLearning.GaussianProcesses import GaussianProcessRegressor
from src.DLL.MachineLearning.SupervisedLearning.Kernels import RBF, Linear, WhiteGaussian, Periodic, RationalQuadratic, Matern
from src.DLL.DeepLearning.Optimisers import ADAM, LBFGS
from src.DLL.Data.Preprocessing import StandardScaler


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
X = torch.linspace(0, 1, 20, dtype=torch.float64, device=device).unsqueeze(1)
Y = torch.sin(3 * 2 * torch.pi * X) + 3 * X ** 2# + torch.randn_like(X) * 0.5
transformer = StandardScaler()
Y = transformer.fit_transform(Y).squeeze(dim=1)

train_kernel = True  # try to changing this line of code to see how the covariance kernel learns the correct parameters

model = GaussianProcessRegressor(Linear(sigma=0.2, sigma_bias=1) ** 2 + Periodic(1, 2, period=0.5), noise=0.1, device=device)
sk_model = GPR(ConstantKernel(constant_value=0.2) * DotProduct(sigma_0=1) ** 2 + ExpSineSquared())
# correlation_length = 0.1
# nu = 0.5
# model = GaussianProcessRegressor(Matern(sigma=2, correlation_length=correlation_length, nu=nu), noise=0.0, device=device)
# sk_model = GPR(sk_matern(nu=nu, length_scale=correlation_length))
model.fit(X, Y)
print(model.log_marginal_likelihood())
if train_kernel:
    # history = model.train_kernel(epochs=2000, optimiser=Adam(), verbose=True)
    history = model.train_kernel(epochs=30, optimiser=LBFGS(model.log_marginal_likelihood, learning_rate=0.1), verbose=True)
    plt.plot(history["log marginal likelihood"])
    plt.xlabel("epoch")
    plt.ylabel("log marginal likelihood")
    plt.title("The change in log marginal likelihood during kernel training")

sk_model.fit(X, Y)
print(sk_model.kernel_.get_params())

x_test = torch.linspace(0, 2, 100, dtype=torch.float64, device=device).unsqueeze(1)
mean, covariance = model.predict(x_test)
mean = mean.squeeze()
mean = transformer.inverse_transform(mean)
covariance = covariance * transformer.var ** 2
std = torch.sqrt(torch.diag(covariance))

plt.figure()
plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
plt.plot(x_test.cpu(), mean.cpu(), color="blue", label="mean")
plt.plot(x_test.cpu(), transformer.inverse_transform(torch.from_numpy(sk_model.predict(x_test.numpy()))), color="lightblue", label="sklearn implementation")
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


try:
    distribution = torch.distributions.MultivariateNormal(mean, covariance)
    plt.figure()
    plt.plot(X.cpu(), transformer.inverse_transform(Y).cpu(), ".")
    for _ in range(5):
        y = distribution.sample()
        plt.plot(x_test.cpu(), y.cpu())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random samples from the previous distribution")
except:
    pass
finally:
    plt.show()
