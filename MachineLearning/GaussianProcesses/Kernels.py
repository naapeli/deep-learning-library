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

class GaussianDistanceCovariance(_Base):
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
    

