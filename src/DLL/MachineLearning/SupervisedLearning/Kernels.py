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
        return self.kernel_1.parameters() | self.kernel_2.parameters()

class RBF(_Base):
    """
    The commonly used radial basis function (rbf) kernel. Yields high values for samples close to one another.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        correlation_length (float, optional): The length scale of the kernel. Determines how quickly the similarity decays as points become further apart. Must be a positive real number. Defaults to 1.
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise ValueError("correlation_length must be a positive real number.")

        RBF.instance += 1
        self.number = RBF.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)

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
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        dists_squared = torch.cdist(X1, X2, p=2) ** 2
        return self(X1, X2) * (dists_squared / (self.correlation_length ** 3))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
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
    The linear kernel, often used as a baseline in kernel-based learning methods, representing a linear relationship between inputs. For the polynomial kernel of degree n, one should use Linear() ** n.

    Args:
        sigma (float, optional): The overall scale factor of the variance. Controls the amplitude of the kernel. Must be a positive real number. Defaults to 1.
        sigma_bias (float, optional): The constant term of the kernel, sometimes called the bias or intercept. It allows the kernel function to handle non-zero means. Must be a real number. Defaults to 0.
    """

    instance = 0
    """
    :meta private:
    """
    def __init__(self, sigma=1, sigma_bias=0):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(sigma_bias, int | float):
            raise TypeError("sigma_bias must be a real number.")

        Linear.instance += 1
        self.number = Linear.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.sigma_bias = torch.tensor([sigma_bias], dtype=torch.float32)

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
        return 2 * self.sigma * X1 @ X2.T

    def derivative_sigma_bias(self, X1, X2):
        """
        :meta private:
        """
        return (2 * self.sigma_bias).to(X1.dtype)

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_sigma_bias = self.derivative_sigma_bias(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
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
    """

    instance = 0
    """
    :meta private:
    """
    def __init__(self, sigma=1):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")

        WhiteGaussian.instance += 1
        self.number = WhiteGaussian.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)

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
        covariance_matrix = (X1[:, None] == X2).all(-1).to(dtype=X1.dtype)
        return 2 * self.sigma * covariance_matrix

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        
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
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1, period=1):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise TypeError("correlation_length must be a real number.")
        if not isinstance(period, int | float) or period <= 0:
            raise TypeError("period must be a real number.")

        Periodic.instance += 1
        self.number = Periodic.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.period = torch.tensor([period], dtype=torch.float32)

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
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        norm = torch.cdist(X1, X2)
        periodic_term = torch.sin(torch.pi * norm / self.period) ** 2
        return 4 * self(X1, X2) * (periodic_term / (self.correlation_length ** 3))

    def derivative_period(self, X1, X2):
        """
        :meta private:
        """
        norm = torch.cdist(X1, X2)
        periodic_term = torch.sin(torch.pi * norm / self.period)
        return 4 * self(X1, X2) * (periodic_term * torch.cos(torch.pi * norm / self.period) * (torch.pi * norm / self.period ** 2) / (self.correlation_length ** 2))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_period = self.derivative_period(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)

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
    """

    instance = 0
    """
    :meta private:
    """

    def __init__(self, sigma=1, correlation_length=1, alpha=1):
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("sigma must be a positive real number.")
        if not isinstance(correlation_length, int | float) or correlation_length <= 0:
            raise TypeError("correlation_length must be a real number.")
        if not isinstance(alpha, int | float) or alpha <= 0:
            raise TypeError("alpha must be a real number.")

        RationalQuadratic.instance += 1
        self.number = RationalQuadratic.instance

        self.sigma = torch.tensor([sigma], dtype=torch.float32)
        self.correlation_length = torch.tensor([correlation_length], dtype=torch.float32)
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

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
        return 2 * self(X1, X2) / self.sigma

    def derivative_corr_len(self, X1, X2):
        """
        :meta private:
        """
        norm_squared = torch.cdist(X1, X2) ** 2
        return (self.sigma ** 2 * 
                (1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)) ** (-self.alpha - 1) *
                (-norm_squared / (self.correlation_length ** 3)))

    def derivative_alpha(self, X1, X2):
        """
        :meta private:
        """
        norm_squared = torch.cdist(X1, X2) ** 2
        term = 1 + norm_squared / (2 * self.alpha * self.correlation_length ** 2)
        return (self(X1, X2) *
                (norm_squared / (2 * self.alpha * self.correlation_length ** 2 + norm_squared) -
                 torch.log(term)))

    def update(self, derivative_function, X, noise=0, epsilon=1e-5):
        """
        :meta private:
        """
        derivative_covariance_sigma = self.derivative_sigma(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_corr_len = self.derivative_corr_len(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)
        derivative_covariance_alpha = self.derivative_alpha(X, X) + (noise + epsilon) * torch.eye(len(X), device=X.device)

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