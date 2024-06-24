import torch


class _Base:
    def __add__(self, other):
        if not isinstance(other, _Base):
            return NotImplemented
        return _Compound(self, other, add=True)
    
    def __mul__(self, other):
        if not isinstance(other, _Base):
            return NotImplemented
        return _Compound(self, other, add=False)
    
    def __pow__(self, power):
        if not isinstance(power, int) or power < 2:
            return NotImplemented
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

class SquaredExponentialCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1):
        self.sigma = sigma
        self.correlation_length = correlation_length

    def __call__(self, data_point_1, data_point_2):
        return self.sigma ** 2 * torch.exp(-(torch.linalg.norm(data_point_2 - data_point_1) ** 2) / (2 * (self.correlation_length ** 2)))

class LinearCovariance(_Base):
    def __init__(self, sigma=1, sigma_bias=0):
        self.sigma = sigma
        self.sigma_bias = sigma_bias

    def __call__(self, data_point_1, data_point_2):
        if len(data_point_1.shape) == 2:
            return self.sigma_bias ** 2 + self.sigma ** 2 * data_point_1.T @ data_point_2
        return self.sigma_bias ** 2 + self.sigma ** 2 * data_point_1 @ data_point_2
    
class WhiteGaussianCovariance(_Base):
    def __init__(self, sigma=1):
        self.sigma = sigma
    
    def __call__(self, data_point_1, data_point_2):
        return (self.sigma * (data_point_1 == data_point_2)).to(dtype=data_point_1.dtype)
    
class OrnsteinUhlenbeckCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1):
        self.sigma = sigma
        self.correlation_length = correlation_length
    
    def __call__(self, data_point_1, data_point_2):
        return self.sigma ** 2 * torch.exp(-torch.linalg.norm(data_point_2 - data_point_1) / self.correlation_length)

class PeriodicCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, period=1):
        self.sigma = sigma
        self.correlation_length = correlation_length
        self.period = period

    def __call__(self, data_point_1, data_point_2):
        return self.sigma ** 2 * torch.exp(-2 * torch.sin(torch.pi * torch.linalg.norm(data_point_2 - data_point_1, ord=1) / self.period) ** 2 / self.correlation_length ** 2)

class RationalQuadraticCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, alpha=1):
        self.sigma = sigma
        self.correlation_length = correlation_length
        self.alpha = alpha
    
    def __call__(self, data_point_1, data_point_2):
        self.sigma ** 2 * (1 + torch.linalg.norm(data_point_2 - data_point_1) ** 2 / (2 * self.alpha * self.correlation_length ** 2)) ** -self.alpha
