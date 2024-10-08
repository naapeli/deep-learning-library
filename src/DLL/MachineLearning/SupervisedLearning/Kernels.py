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

    def __call__(self, X1, X2):
        if self.add:
            return self.kernel_1(X1, X2) + self.kernel_2(X1, X2)
        else:
            return self.kernel_1(X1, X2) * self.kernel_2(X1, X2)

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        if self.add:
            self.kernel_1.update(derivative_function, X, noise=noise, epsilon=epsilon)
            self.kernel_2.update(derivative_function, X, noise=noise, epsilon=epsilon)
        else:
            kernel_1_covariance = self.kernel_1(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
            kernel_2_covariance = self.kernel_2(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)

            kernel_1_derivative = lambda parameter_derivative: derivative_function(parameter_derivative @ kernel_2_covariance)
            kernel_2_derivative = lambda parameter_derivative: derivative_function(kernel_1_covariance @ parameter_derivative)

            self.kernel_1.update(kernel_1_derivative, X, noise=0, epsilon=epsilon)
            self.kernel_2.update(kernel_2_derivative, X, noise=0, epsilon=epsilon)

    def parameters(self):
        return [*self.kernel_1.parameters(), *self.kernel_2.parameters()]

class SquaredExponentialCovariance(_Base):
    def __init__(self, sigma=1.0, correlation_length=1.0, device=torch.device("cpu")):
        self.sigma = torch.tensor([sigma], dtype=torch.float32, device=device)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32, device=device)

    def __call__(self, X1, X2):
        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(X1.device)
        
        dists_squared = torch.cdist(X1, X2, p=2) ** 2
        return self.sigma ** 2 * torch.exp(-dists_squared / (2 * (self.correlation_length ** 2)))

    def derivative_sigma(self, X1, X2):
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        dists_squared = torch.cdist(X1, X2, p=2) ** 2
        return self(X1, X2) * (dists_squared / (self.correlation_length ** 3))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)

    def parameters(self):
        return [self.sigma, self.correlation_length]

class LinearCovariance(_Base):
    def __init__(self, sigma=1, sigma_bias=0):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.sigma_bias = torch.tensor([sigma_bias], dtype=torch.float32)

    def __call__(self, X1, X2):
        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.sigma_bias = self.sigma_bias.to(X1.device)
        
        return self.sigma_bias ** 2 + self.sigma ** 2 * X1 @ X2.T

    def derivative_sigma(self, X1, X2):
        return 2 * self.sigma * X1 @ X2.T

    def derivative_sigma_bias(self, X1, X2):
        return (2 * self.sigma_bias).to(X1.dtype)

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_sigma_bias = self.derivative_sigma_bias(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.sigma_bias.grad = derivative_function(derivative_covariance_sigma_bias).to(dtype=self.sigma_bias.dtype).squeeze(0)

    def parameters(self):
        return [self.sigma, self.sigma_bias]

class WhiteGaussianCovariance(_Base):
    def __init__(self, sigma=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)

    def __call__(self, X1, X2):
        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
        
        # Create an identity-like matrix using a vectorized equality check
        covariance_matrix = (X1[:, None] == X2).all(-1).to(dtype=X1.dtype)
        
        # Multiply by sigma^2
        return self.sigma ** 2 * covariance_matrix

    def derivative_sigma(self, X1, X2):
        # Derivative of the covariance matrix with respect to sigma
        covariance_matrix = (X1[:, None] == X2).all(-1).to(dtype=X1.dtype)
        return 2 * self.sigma * covariance_matrix

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        # Compute the derivative covariance matrix in a vectorized way
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
        # Update the gradient of sigma
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)

    def parameters(self):
        return [self.sigma]

class PeriodicCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, period=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.period = torch.tensor([period], dtype=torch.float32)

    def __call__(self, X1, X2):
        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(device=X1.device)
            self.period = self.period.to(device=X1.device)
        
        # Calculate the periodic covariance matrix in a vectorized way
        norm = torch.cdist(X1, X2, p=1)  # L1 distance (Manhattan distance) between points
        periodic_term = torch.sin(torch.pi * norm / self.period) ** 2
        covariance_matrix = self.sigma ** 2 * torch.exp(-2 * periodic_term / (self.correlation_length ** 2))
        return covariance_matrix

    def derivative_sigma(self, X1, X2):
        # Derivative with respect to sigma
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        norm = torch.cdist(X1, X2, p=1)
        periodic_term = torch.sin(torch.pi * norm / self.period) ** 2
        return 4 * self(X1, X2) * (periodic_term / (self.correlation_length ** 3))

    def derivative_period(self, X1, X2):
        norm = torch.cdist(X1, X2, p=1)
        periodic_term = torch.sin(torch.pi * norm / self.period)
        return 4 * self(X1, X2) * (periodic_term * torch.cos(torch.pi * norm / self.period) * (torch.pi * norm / self.period ** 2) / (self.correlation_length ** 2))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        # Vectorized computation of derivative covariance matrices
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_period = self.derivative_period(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)

        # Update gradients for sigma, correlation length, and period
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.period.grad = derivative_function(derivative_covariance_period).to(dtype=self.period.dtype).squeeze(0)

    def parameters(self):
        return [self.sigma, self.correlation_length, self.period]

class RationalQuadraticCovariance(_Base):
    def __init__(self, sigma=1, correlation_length=1, alpha=1):
        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def __call__(self, X1, X2):
        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(X1.device)
            self.alpha = self.alpha.to(X1.device)
        
        norm_squared = torch.cdist(X1, X2) ** 2
        return self.sigma ** 2 * (1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)) ** -self.alpha

    def derivative_sigma(self, X1, X2):
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        norm_squared = torch.cdist(X1, X2) ** 2
        return (self.sigma ** 2 * 
                (1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)) ** (-self.alpha - 1) *
                (-norm_squared / (self.correlation_length ** 3)))

    def derivative_alpha(self, X1, X2):
        norm_squared = torch.cdist(X1, X2) ** 2
        term = 1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)
        return (self(X1, X2) *
                (norm_squared / (2 * self.alpha * self.correlation_length ** 2 + norm_squared) -
                 torch.log(term)))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_alpha = self.derivative_alpha(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)

        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.alpha.grad = derivative_function(derivative_covariance_alpha).to(dtype=self.alpha.dtype).squeeze(0)

    def parameters(self):
        return [self.sigma, self.correlation_length, self.alpha]