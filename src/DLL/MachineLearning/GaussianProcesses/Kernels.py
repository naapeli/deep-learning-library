import torch


class _Base:
    def __add__(self, other):
        if not isinstance(other, _Base):
            raise NotImplementedError()
        return _Compound(self, other, add=True)
    
    def __mul__(self, other):
        if not isinstance(other, _Base):
            raise NotImplementedError()
        return _Compound(self, other, add=False)
    
    def __pow__(self, power):
        if not isinstance(power, int) or power < 2:
            raise NotImplementedError()
        kernel = _Compound(self, self, add=False)
        for _ in range(power - 1):
            kernel = _Compound(kernel, self, add=False)
        return kernel

class _Compound(_Base):
    def __init__(self, kernel_1, kernel_2, add):
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.add = add

    def __call__(self, data_point_1, data_point_2):
        if self.add:
            return self.kernel_1(data_point_1, data_point_2) + self.kernel_2(data_point_1, data_point_2)
        else:
            return self.kernel_1(data_point_1, data_point_2) * self.kernel_2(data_point_1, data_point_2)
    
    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        if self.add:
            self.kernel_1.update(derivative_function, X, noise=noise, epsilon=epsilon)
            self.kernel_2.update(derivative_function, X, noise=noise, epsilon=epsilon)
        else:
            kernel_1_covariance = torch.tensor([[self.kernel_1(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=X.device).T + (noise + epsilon) * torch.eye(len(X), device=X.device)
            kernel_2_covariance = torch.tensor([[self.kernel_2(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=X.device).T + (noise + epsilon) * torch.eye(len(X), device=X.device)
            kernel_1_derivative = lambda parameter_derivative: derivative_function(parameter_derivative @ kernel_2_covariance)
            kernel_2_derivative = lambda parameter_derivative: derivative_function(kernel_1_covariance @ parameter_derivative)
            self.kernel_1.update(kernel_1_derivative, X, noise=0, epsilon=1e-5)
            self.kernel_2.update(kernel_2_derivative, X, noise=0, epsilon=1e-5)
    
    def parameters(self):
        return [*self.kernel_1.parameters(), *self.kernel_2.parameters()]

class SquaredExponentialCovariance(_Base):
    def __init__(self, sigma=1.0, correlation_length=1.0, device=torch.device("cpu")):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)

    def __call__(self, data_point_1, data_point_2):
        if self.sigma.device != data_point_1.device:
            self.sigma = self.sigma.to(data_point_1.device)
            self.correlation_length = self.correlation_length.to(device=data_point_1.device)
        return self.sigma ** 2 * torch.exp(-(torch.linalg.norm(data_point_2 - data_point_1) ** 2) / (2 * (self.correlation_length ** 2)))
    
    def derivative_sigma(self, data_point_1, data_point_2):
        return 2 * self(data_point_1, data_point_2) / self.sigma
    
    def derivative_corr_len(self, data_point_1, data_point_2):
        return self(data_point_1, data_point_2) * ((torch.linalg.norm(data_point_2 - data_point_1) ** 2) / (self.correlation_length ** 3))
    
    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = torch.tensor([[self.derivative_sigma(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_corr_len = torch.tensor([[self.derivative_corr_len(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
    
    def parameters(self):
        return [self.sigma, self.correlation_length]

class LinearCovariance(_Base):
    def __init__(self, sigma=1, sigma_bias=0):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.sigma_bias = torch.tensor([sigma_bias], dtype=torch.float32)

    def __call__(self, data_point_1, data_point_2):
        if self.sigma.device != data_point_1.device:
            self.sigma = self.sigma.to(data_point_1.device)
            self.sigma_bias = self.sigma_bias.to(device=data_point_1.device)
        if len(data_point_1.shape) == 2:
            return self.sigma_bias ** 2 + self.sigma ** 2 * data_point_1.T @ data_point_2
        return self.sigma_bias ** 2 + self.sigma ** 2 * data_point_1 @ data_point_2
    
    def derivative_sigma(self, data_point_1, data_point_2):
        if len(data_point_1.shape) == 2:
            return 2 * self.sigma * data_point_1.T @ data_point_2
        return 2 * self.sigma * data_point_1 @ data_point_2
    
    def derivative_sigma_bias(self, data_point_1, data_point_2):
        return 2 * self.sigma_bias
    
    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = torch.tensor([[self.derivative_sigma(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_sigma_bias = torch.tensor([[self.derivative_sigma_bias(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.sigma_bias.grad = derivative_function(derivative_covariance_sigma_bias).to(dtype=self.sigma_bias.dtype).squeeze(0)
    
    def parameters(self):
        return [self.sigma, self.sigma_bias]
    
class WhiteGaussianCovariance(_Base):
    def __init__(self, sigma=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
    
    def __call__(self, data_point_1, data_point_2):
        if self.sigma.device != data_point_1.device:
            self.sigma = self.sigma.to(data_point_1.device)
        return (self.sigma ** 2 * (data_point_1 == data_point_2)).to(dtype=data_point_1.dtype)
    
    def derivative_sigma(self, data_point_1, data_point_2):
        return (2 * self.sigma * (data_point_1 == data_point_2)).to(dtype=data_point_1.dtype)
    
    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = torch.tensor([[self.derivative_sigma(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
    
    def parameters(self):
        return [self.sigma]

# Log-marginal-likelihood doesn't decrease when only considering correlation length or period (derivatives may be wrong, even though they are calculated with Mathematica)
class PeriodicCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, period=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.period = torch.tensor([period], dtype=torch.float32)

    def __call__(self, data_point_1, data_point_2):
        if self.sigma.device != data_point_1.device:
            self.sigma = self.sigma.to(data_point_1.device)
            self.correlation_length = self.correlation_length.to(device=data_point_1.device)
            self.period = self.period.to(device=data_point_1.device)
        return self.sigma ** 2 * torch.exp(-2 * torch.sin(torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period) ** 2 / self.correlation_length ** 2)
    
    # derivatives with Mathematica
    def derivative_sigma(self, data_point_1, data_point_2):
        return 2 * self(data_point_1, data_point_2) / self.sigma
    
    def derivative_corr_len(self, data_point_1, data_point_2):
        return 4 * self(data_point_1, data_point_2) * (torch.sin(torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period) ** 2 / self.correlation_length ** 3)
    
    def derivative_period(self, data_point_1, data_point_2):
        return 4 * self(data_point_1, data_point_2) * (torch.sin(torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period) * torch.cos(torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period) * (torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period ** 2) / self.correlation_length ** 2)

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = torch.tensor([[self.derivative_sigma(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_corr_len = torch.tensor([[self.derivative_corr_len(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_period = torch.tensor([[self.derivative_period(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.period.grad = derivative_function(derivative_covariance_period).to(dtype=self.period.dtype).squeeze(0)
    
    def parameters(self):
        return [self.sigma, self.correlation_length, self.period]

# Log-marginal-likelihood doesn't decrease when only considering correlation length or alpha (derivatives may be wrong, even though they are calculated with Mathematica)
class RationalQuadraticCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, alpha=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.alpha = torch.tensor([alpha], dtype=torch.float32)
    
    def __call__(self, data_point_1, data_point_2):
        if self.sigma.device != data_point_1.device:
            self.sigma = self.sigma.to(data_point_1.device)
            self.correlation_length = self.correlation_length.to(device=data_point_1.device)
            self.alpha = self.alpha.to(device=data_point_1.device)
        return self.sigma ** 2 * (1 + torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (2 * self.alpha * self.correlation_length ** 2)) ** -self.alpha

    # derivatives with Mathematica
    def derivative_sigma(self, data_point_1, data_point_2):
        return 2 * self(data_point_1, data_point_2) / self.sigma
    
    def derivative_corr_len(self, data_point_1, data_point_2):
        return self.sigma ** 2 * ((1 + torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (2 * self.alpha * self.correlation_length ** 2)) ** (-self.alpha - 1)) * (-torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (self.correlation_length ** 3))
    
    def derivative_alpha(self, data_point_1, data_point_2):
        return self(data_point_1, data_point_2) * (torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (2 * self.alpha * self.correlation_length ** 2 + torch.linalg.norm(data_point_2 - data_point_1) ** 2) - torch.log(1 + torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (2 * self.alpha * self.correlation_length ** 2)))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = torch.tensor([[self.derivative_sigma(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_corr_len = torch.tensor([[self.derivative_corr_len(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        derivative_covariance_alpha = torch.tensor([[self.derivative_alpha(x1, x2) for x1 in X] for x2 in X], dtype=X.dtype, device=self.sigma.device).T + (noise + epsilon) * torch.eye(len(X), device=self.sigma.device)
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.alpha.grad = derivative_function(derivative_covariance_alpha).to(dtype=self.alpha.dtype).squeeze(0)
    
    def parameters(self):
        return [self.sigma, self.correlation_length, self.alpha]
