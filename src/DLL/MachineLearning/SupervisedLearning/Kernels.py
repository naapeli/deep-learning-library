import torch
from scipy.special import gamma, kv
from math import sqrt


class _Base:
    def __add__(self, other):
        if not isinstance(other, _Base):
            raise NotImplementedError()
        return _Compound(self, other, add=True)
    
    def __mul__(self, other):
        if not isinstance(other, _Base):
            raise NotImplementedError()
        return _Compound(self, other, multiply=True)
    
    def __pow__(self, power):
        if not isinstance(power, int) or power < 2:
            raise ValueError("The exponent must be an integer greater than 1.")
        kernel = _Exponent(self, power=power)
        return kernel

class _Compound(_Base):
    def __init__(self, kernel_1, kernel_2, add=False, multiply=False):
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.add = add
        self.multiply = multiply

    def __call__(self, X1, X2):
        if self.add:
            return self.kernel_1(X1, X2) + self.kernel_2(X1, X2)
        elif self.multiply:
            return self.kernel_1(X1, X2) * self.kernel_2(X1, X2)

    def update(self, derivative_function, X):
        if self.add:
            self.kernel_1.update(derivative_function, X)
            self.kernel_2.update(derivative_function, X)
        elif self.multiply:
            kernel_1_covariance = self.kernel_1(X, X)
            kernel_2_covariance = self.kernel_2(X, X)

            kernel_1_derivative = lambda parameter_derivative: derivative_function(parameter_derivative * kernel_2_covariance)
            kernel_2_derivative = lambda parameter_derivative: derivative_function(kernel_1_covariance * parameter_derivative)

            self.kernel_1.update(kernel_1_derivative, X)
            self.kernel_2.update(kernel_2_derivative, X)

    def parameters(self):
        return self.kernel_1.parameters() | self.kernel_2.parameters()

class _Exponent(_Base):
    def __init__(self, kernel, power):
        self.kernel = kernel
        self.power = power
    
    def __call__(self, X1, X2):
        return self.kernel(X1, X2) ** self.power
    
    def update(self, derivative_function, X):
        kernel_derivative = lambda parameter_derivative: derivative_function(self.power * self.kernel(X, X) ** (self.power - 1) * parameter_derivative)
        self.kernel.update(kernel_derivative, X)
    
    def parameters(self):
        return self.kernel.parameters()

class RBF(_Base):
    """
    The commonly used radial basis function (rbf) kernel. Yields high values for samples close to one another.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        correlation_length (float, optional): The length scale of the kernel. Determines how quickly the similarity decays as points become further apart. Must be a positive real number. Defaults to 1.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
        train_correlation_length (bool, optional): Determines, wheter or not the correlation_length parameter should be changed during training the kernel. Defaults to True.
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1, train_sigma=True, train_correlation_length=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise ValueError("correlation_length must be a positive real number.")
        if not isinstance(train_sigma, bool) or not isinstance(train_correlation_length, bool):
            raise TypeError("train_sigma andtrain_correlation_length must be boolean values. ")

        RBF.instance += 1
        self.number = RBF.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)

        self.train_sigma = train_sigma
        self.train_correlation_length = train_correlation_length

    def __call__(self, X1, X2):
        """
        :meta public:

        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))

        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(X1.device)
        
        dists_squared = torch.cdist(X1, X2, p=2) ** 2
        return self.sigma ** 2 * torch.exp(-dists_squared / (2 * (self.correlation_length ** 2)))

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = 2 * self(X1, X2) / self.sigma
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_correlation_length: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        dists_squared = torch.cdist(X1, X2, p=2) ** 2
        val = self(X1, X2) * (dists_squared / (self.correlation_length ** 3))
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.correlation_length += eps
        # descrete += self(X1, X2)
        # self.correlation_length -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X)
        
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "rbf_sigma_1".
        """
        return {("rbf_sigma" + "_" + str(self.number)): self.sigma, ("rbf_corr_len" + "_" + str(self.number)): self.correlation_length}

class Linear(_Base):
    """
    The linear kernel, often used as a baseline in kernel-based learning methods, representing a linear relationship between inputs.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        sigma_bias (float, optional): The constant term of the kernel, sometimes called the bias or intercept. It allows the kernel function to handle non-zero means. Must be a real number. Defaults to 0.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
        train_sigma_bias (bool, optional): Determines, wheter or not the sigma_bias parameter should be changed during training the kernel. Defaults to True.
        
    Example:
        The commonly used polynomial kernel can be used as follows:

        .. code-block:: python
        
            from DLL.MachineLearning.SupervisedLearning.Kernels import Linear

            linear_kernel = Linear()
            polynomial_kernel_degree_d = linear_kernel ** d
    """

    instance = 0
    """
    :meta private:
    """
    def __init__(self, sigma=1, sigma_bias=0, train_sigma=True, train_sigma_bias=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(sigma_bias, int | float):
            raise TypeError("sigma_bias must be a real number.")
        if not isinstance(train_sigma, bool) or not isinstance(train_sigma_bias, bool):
            raise TypeError("train_sigma and train_sigma_bias must be boolean values.")

        Linear.instance += 1
        self.number = Linear.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.sigma_bias = torch.tensor([sigma_bias], dtype=torch.float32)

        self.train_sigma = train_sigma
        self.train_sigma_bias = train_sigma_bias

    def __call__(self, X1, X2):
        """
        :meta public:

        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))

        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.sigma_bias = self.sigma_bias.to(X1.device)
        
        return self.sigma_bias ** 2 + self.sigma ** 2 * X1 @ X2.T

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = 2 * self.sigma * X1 @ X2.T
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[10] / eps, val[10])
        return val

    def derivative_sigma_bias(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma_bias: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = (2 * self.sigma_bias).to(X1.dtype)
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma_bias += eps
        # descrete += self(X1, X2)
        # self.sigma_bias -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        derivative_covariance_sigma_bias = self.derivative_sigma_bias(X, X)
        
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.sigma_bias.grad = derivative_function(derivative_covariance_sigma_bias).to(dtype=self.sigma_bias.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "linear_sigma_1".
        """
        return {("linear_sigma" + "_" + str(self.number)): self.sigma, ("linear_sigma_bias" + "_" + str(self.number)): self.sigma_bias}

class WhiteGaussian(_Base):
    """
    The white Gaussian kernel, commonly used to capture Gaussian noise in data. This kernel models purely random noise without dependencies on input values.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
    """

    instance = 0
    """
    :meta private:
    """
    def __init__(self, sigma=1, train_sigma=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(train_sigma, bool):
            raise TypeError("train_sigma must be a boolean value.")

        WhiteGaussian.instance += 1
        self.number = WhiteGaussian.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.train_sigma = train_sigma

    def __call__(self, X1, X2):
        """
        :meta public:

        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))
        
        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
        
        covariance_matrix = (X1[:, None] == X2).all(-1).to(dtype=X1.dtype)
        return self.sigma ** 2 * covariance_matrix

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        covariance_matrix = (X1[:, None] == X2).all(-1).to(dtype=X1.dtype)
        val = 2 * self.sigma * covariance_matrix
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        
        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "white_gaussian_sigma_1".
        """
        return {("white_gaussian_sigma" + "_" + str(self.number)): self.sigma}

class Periodic(_Base):
    """
    The periodic kernel, commonly used to capture periodic relationships in data, such as seasonal patterns or repeating cycles.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        correlation_length (float, optional): Controls how quickly the similarity decays as points move further apart in the input space. Must be a positive real number. Defaults to 1.
        period (float, optional): The period of the kernel, indicating the distance over which the function repeats. Must be a positive real number. Defaults to 1.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
        train_correlation_length (bool, optional): Determines, wheter or not the correlation_length parameter should be changed during training the kernel. Defaults to True.
        train_period (bool, optional): Determines, wheter or not the period parameter should be changed during training the kernel. Defaults to True.
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1, period=1, train_sigma=True, train_correlation_length=True, train_period=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise TypeError("correlation_length must be a real number.")
        if not isinstance(period, int | float) or period <= 0:
            raise TypeError("period must be a real number.")
        if not isinstance(train_sigma, bool) or not isinstance(train_correlation_length, bool) or not isinstance(train_period, bool):
            raise TypeError("train_sigma, train_correlation_length and train_period must be boolean values.")
        

        Periodic.instance += 1
        self.number = Periodic.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.period = torch.tensor([period], dtype=torch.float32)

        self.train_sigma = train_sigma
        self.train_correlation_length = train_correlation_length
        self.train_period = train_period

    def __call__(self, X1, X2):
        """
        :meta public:

        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))
        
        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(device=X1.device)
            self.period = self.period.to(device=X1.device)
        
        norm = torch.cdist(X1, X2)
        periodic_term = torch.sin(torch.pi * norm / self.period) ** 2
        covariance_matrix = self.sigma ** 2 * torch.exp(-2 * periodic_term / (self.correlation_length ** 2))
        return covariance_matrix

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = 2 * self(X1, X2) / self.sigma
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_correlation_length: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        norm = torch.cdist(X1, X2)
        periodic_term = torch.sin(torch.pi * norm / self.period) ** 2
        val = 4 * self(X1, X2) * (periodic_term / (self.correlation_length ** 3))
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.correlation_length += eps
        # descrete += self(X1, X2)
        # self.correlation_length -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def derivative_period(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_period: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        norm = torch.cdist(X1, X2)
        periodic_term = torch.sin(torch.pi * norm / self.period)
        val = 4 * self(X1, X2) * (periodic_term * torch.cos(torch.pi * norm / self.period) * (torch.pi * norm / self.period ** 2) / (self.correlation_length ** 2))
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.period += eps
        # descrete += self(X1, X2)
        # self.period -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X)
        derivative_covariance_period = self.derivative_period(X, X)

        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.period.grad = derivative_function(derivative_covariance_period).to(dtype=self.period.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "periodic_sigma_1".
        """
        return {("periodic_sigma" + "_" + str(self.number)): self.sigma, ("periodic_corr_len" + "_" + str(self.number)): self.correlation_length, ("periodic_period" + "_" + str(self.number)): self.period}

class RationalQuadratic(_Base):
    """
    The rational quadratic kernel, a versatile kernel often used in Gaussian Processes for modeling data with varying degrees of smoothness. It can be seen as a scale mixture of the squared exponential kernel, allowing flexibility between linear and non-linear relationships.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        correlation_length (float, optional): Controls how quickly the similarity decays as points move further apart in the input space. Must be a positive real number. Defaults to 1.
        alpha (float, optional): Controls the relative weighting of large-scale and small-scale variations. Higher values make the kernel behave more like a squared exponential (Gaussian) kernel, while lower values allow for more flexibility. Must be a positive real number. Defaults to 1.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
        train_correlation_length (bool, optional): Determines, wheter or not the correlation_length parameter should be changed during training the kernel. Defaults to True.
        train_alpha (bool, optional): Determines, wheter or not the alpha parameter should be changed during training the kernel. Defaults to True.
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1, alpha=1, train_sigma=True, train_correlation_length=True, train_alpha=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise TypeError("correlation_length must be a real number.")
        if not isinstance(alpha, int | float) or alpha <= 0:
            raise TypeError("alpha must be a real number.")
        if not isinstance(train_sigma, bool) or not isinstance(train_correlation_length, bool) or not isinstance(train_alpha, bool):
            raise TypeError("train_sigma, train_correlation_length and train_alpha must be boolean values.")

        RationalQuadratic.instance += 1
        self.number = RationalQuadratic.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

        self.train_sigma = train_sigma
        self.train_correlation_length = train_correlation_length
        self.train_alpha = train_alpha

    def __call__(self, X1, X2):
        """
        :meta public:
        
        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))
        
        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(X1.device)
            self.alpha = self.alpha.to(X1.device)
        
        norm_squared = torch.cdist(X1, X2) ** 2
        return self.sigma ** 2 * (1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)) ** -self.alpha

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = 2 * self(X1, X2) / self.sigma
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_correlation_length: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        norm_squared = torch.cdist(X1, X2) ** 2
        val = (self.sigma ** 2 * 
                (1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)) ** (-self.alpha - 1) *
                (norm_squared / (self.correlation_length ** 3)))
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.correlation_length += eps
        # descrete += self(X1, X2)
        # self.correlation_length -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def derivative_alpha(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_alpha: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        norm_squared = torch.cdist(X1, X2) ** 2
        term = 1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)
        val = (self(X1, X2) *
                (norm_squared / (2 * self.alpha * self.correlation_length ** 2 + norm_squared) -
                 torch.log(term)))
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.alpha += eps
        # descrete += self(X1, X2)
        # self.alpha -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X)
        derivative_covariance_alpha = self.derivative_alpha(X, X)

        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)
        self.alpha.grad = derivative_function(derivative_covariance_alpha).to(dtype=self.alpha.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "rational_quadratic_sigma_1".
        """
        return {("rational_quadratic_sigma" + "_" + str(self.number)): self.sigma, ("rational_quadratic_corr_len" + "_" + str(self.number)): self.correlation_length, ("rational_quadratic_alpha" + "_" + str(self.number)): self.alpha}
    
class Matern(_Base):
    """
    The Matern kernel, a versatile kernel often used in Gaussian Processes for modeling data with varying degrees of smoothness. Is a generalization of the RBF kernel with varying levels of smoothness controlled by nu.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        correlation_length (float, optional): Controls how quickly the similarity decays as points move further apart in the input space. Must be a positive real number. Defaults to 1.
        nu (float, optional): Controls the smoothness of the kernel. Important values of nu include 0.5 for the rbf kernel with the l1 norm modelling non differentiable functions, 1.5 for once differentiable functions and 2.5 for twice differentiable functions. If is set to float("inf"), the kernel is equivalent to the RBF kernel. It is not possible to train nu. Must be a positive real number. Defaults to 1.5.
        train_sigma (bool, optional): Determines, wheter or not the sigma parameter should be changed during training the kernel. Defaults to True.
        train_correlation_length (bool, optional): Determines, wheter or not the correlation_length parameter should be changed during training the kernel. Defaults to True.
    """

    instance = 0
    """
    :meta private:
    """

    def __new__(cls, sigma=1, correlation_length=1, nu=1.5, train_sigma=True, train_correlation_length=True):
        # if nu is infinite, the kernel is equivalent to the RBF kernel
        if nu == float("inf"):
            return RBF(sigma, correlation_length, train_sigma, train_correlation_length)
        return super().__new__(cls)

    def __init__(self, sigma=1, correlation_length=1, nu=1, zero_d_eps=1e-5, train_sigma=True, train_correlation_length=True):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise TypeError("correlation_length must be a real number.")
        if not isinstance(nu, int | float) or nu <= 0:
            raise TypeError("nu must be a real number.")
        if not isinstance(train_sigma, bool) or not isinstance(train_correlation_length, bool):
            raise TypeError("train_sigma and train_correlation_length must be boolean values.")

        RationalQuadratic.instance += 1
        self.number = RationalQuadratic.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.nu = torch.tensor([nu], dtype=torch.float32)

        self.zero_d_eps = zero_d_eps
        self._gamma_val = gamma(nu)

        self.train_sigma = train_sigma
        self.train_correlation_length = train_correlation_length

    def __call__(self, X1, X2):
        """
        :meta public:
        
        Yields the kernel matrix between two vectors.

        Args:
            X1 (torch.Tensor of shape (n_samples_1, n_features))
            X2 (torch.Tensor of shape (n_samples_2, n_features))
        
        Returns:
            kernel_matrix (torch.Tensor of shape (n_samples_1, n_samples_2)): The pairwise kernel values between samples from X1 and X2.
        
        Raises:
            TypeError: If the input matricies are not a PyTorch tensors.
            ValueError: If the input matricies are not the correct shape.
        """
        if not isinstance(X1, torch.Tensor) or not isinstance(X2, torch.Tensor):
            raise TypeError("The input matricies must be PyTorch tensors.")
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("The input matricies must be 2 dimensional tensors.")

        if self.sigma.device != X1.device:
            self.sigma = self.sigma.to(X1.device)
            self.correlation_length = self.correlation_length.to(X1.device)
            self.nu = self.nu.to(X1.device)
        
        norm = torch.cdist(X1, X2)
        y = torch.sqrt(2 * self.nu) * norm / self.correlation_length + self.zero_d_eps
        bessel_factor = kv(self.nu, y)
        constant_factor = 2 ** (1 - self.nu) / self._gamma_val
        other_factor = y ** self.nu
        return self.sigma ** 2 * constant_factor * other_factor * bessel_factor

    def derivative_sigma(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_sigma: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        val = 2 * self(X1, X2) / self.sigma
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.sigma += eps
        # descrete += self(X1, X2)
        # self.sigma -= eps
        # print(descrete[0] / eps, val[0])
        return val
    
    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        if not self.train_correlation_length: return torch.zeros((len(X1), len(X2)), dtype=X1.dtype, device=X1.device)
        #  derivatives from wolfram mathematica
        norm = torch.cdist(X1, X2)
        y = torch.sqrt(2 * self.nu) * norm / self.correlation_length + self.zero_d_eps
        val = -((2 ** (-self.nu / 2) * (norm * torch.sqrt(self.nu) / self.correlation_length) ** self.nu * torch.sqrt(self.nu) * self.sigma ** 2 *
            (-sqrt(2) * norm * kv(self.nu - 1, y) + 2 * self.correlation_length * kv(self.nu, y) - sqrt(2) * norm * kv(self.nu + 1, y))
        )) / (self.correlation_length ** 2 * self._gamma_val)
        # descrete = -self(X1, X2)
        # eps = 1e-6
        # self.correlation_length += eps
        # descrete += self(X1, X2)
        # self.correlation_length -= eps
        # print(descrete[0] / eps, val[0])
        return val

    def update(self, derivative_function, X):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X)

        self.sigma.grad = derivative_function(derivative_covariance_sigma).to(dtype=self.sigma.dtype).squeeze(0)
        self.correlation_length.grad = derivative_function(derivative_covariance_corr_len).to(dtype=self.correlation_length.dtype).squeeze(0)

    def parameters(self):
        """
        Yields the parameters of the kernel as a dictionary. If one uses a combination of the kernels, the parameters of each of the child kernels are returned.

        Returns:
            parameters (dict[str, torch.Tensor]): The parameters as a dictionary. The key of the parameter is eg. "matern_sigma_1".
        """
        return {("matern_sigma" + "_" + str(self.number)): self.sigma, ("matern_corr_len" + "_" + str(self.number)): self.correlation_length}
    
