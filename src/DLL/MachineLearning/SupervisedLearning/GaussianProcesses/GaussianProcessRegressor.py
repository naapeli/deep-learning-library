import torch
from math import floor

from ..Kernels import _Base
from ....DeepLearning.Optimisers import Adam
from ....DeepLearning.Optimisers.BaseOptimiser import BaseOptimiser
from ....Exceptions import NotFittedError
from ....Data.Metrics import _round_dictionary


class GaussianProcessRegressor:
    """
    Implements the Gaussian process regression model.

    Args:
        covariance_function (:ref:`kernel_section_label`, optional): The kernel function expressing how similar are different samples.
        noise (int | float, optional): The artificially added noise to the model. Is added as variance to each sample. Must be non-negative. Defaults to 0.
        epsilon (float, optional): Implemented similarly to noise. Makes sure the covariance matrix is positive definite and hence invertible. Must be positive. Defaults to 1e-5. If one gets a RunTimeError for a matrix not being invertible, one should increase this parameter.
        device (torch.device, optional): The device of all matricies. Defaults to torch.device("cpu").
        
    Attributes:
        n_features (int): The number of features. Available after fitting.
    """
    def __init__(self, covariance_function, noise=0, epsilon=1e-5, device=torch.device("cpu")):
        if not isinstance(covariance_function, _Base):
            raise TypeError("covariance_function must be from DLL.MachineLearning.Supervisedlearning.Kernels.")
        if not isinstance(noise, int | float) or noise < 0:
            raise ValueError("noise must b non-negative.")
        if not isinstance(epsilon, int | float) or epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not isinstance(device, torch.device):
            raise TypeError("device must be an instance of torch.device.")

        self.covariance_function = covariance_function
        self.noise = noise
        self.epsilon = epsilon
        self.device = device

    def _get_covariance_matrix(self, X1, X2):
        return self.covariance_function(X1, X2).to(X1.dtype).to(self.device)

    def fit(self, X, y):
        """
        Fits the GaussianProcessRegressor model to the input data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample. Should be normalized to zero mean and one variance.
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the target matrix is not a PyTorch tensor.
            ValueError: If the input matrix or the target matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")
        if len(y) < 2:
            raise ValueError("There must be atleast 2 samples.")
        # variance = torch.var(y)
        # mean = torch.mean(y)
        # if not torch.allclose(variance, torch.ones_like(variance)):
        #     raise ValueError("y must have one variance. Use DLL.Data.Preprocessing.StandardScaler to scale the data.")
        # if not torch.allclose(mean, torch.zeros_like(mean)):
        #     raise ValueError("y must have zero mean. Use DLL.Data.Preprocessing.StandardScaler to scale the data.")

        self.n_features = X.shape[1]
        self.X = X
        self.Y = y.unsqueeze(dim=1)
        self.prior_covariance_matrix = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X), device=self.device)
        self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)

    def predict(self, X):
        """
        Applies the fitted GaussianProcessRegressor model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            mean, covariance (tuple[torch.Tensor of shape (n_samples,), torch.Tensor of shape (n_samples, n_samples)): A tuple containing the posterior mean and posterior covariance. As the prediction, one should use the mean.
        Raises:
            NotFittedError: If the GaussianProcessRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "inverse_prior_covariance_matrix"):
            raise NotFittedError("GaussianProcessRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        k_1 = self._get_covariance_matrix(self.X, X)
        k_2 = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X), device=self.device)
        mean = k_1.T @ self.inverse_prior_covariance_matrix @ self.Y
        posterior_covariance = k_2 - k_1.T @ self.inverse_prior_covariance_matrix @ k_1
        return mean, posterior_covariance
    
    def log_marginal_likelihood(self):
        """
        Computes the log marginal likelihood of the current model. This value is used to optimize hyperparameters.

        Returns:
            log marginal likelihood (float): The log marginal likelihood of the current model.
        """
        if not hasattr(self, "inverse_prior_covariance_matrix"):
            raise NotFittedError("GaussianProcessRegressor.fit() must be called before calculating the log marginal likelihood.")

        L = torch.linalg.cholesky(self.prior_covariance_matrix)
        alpha = torch.cholesky_solve(self.Y, L)
        lml = -0.5 * self.Y.T @ alpha - torch.sum(torch.log(torch.diagonal(L))) - 0.5 * len(self.Y) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        return lml.item()
    
    def train_kernel(self, epochs=10, optimiser=None, callback_frequency=1, verbose=False):
        """
        Trains the current covariance function parameters by maximizing the log marginal likelihood.

        Args:
            epochs (int, optional): The number of optimisation rounds. Must be a positive integer. Defaults to 10.
            optimiser (:ref:`optimisers_section_label` | None, optional): The optimiser used for training the model. If None, the Adam optimiser is used.
            callback_frequency (int, optional): The number of iterations between printing info from training. Must be a positive integer. Defaults to 1, which means that every iteration, info is printed assuming verbose=True.
            verbose (bool, optional): If True, prints the log marginal likelihood of the model during training. Defaults to False.
        
        Returns:
            history (dict[str, torch.Tensor], the tensor is floor(epochs / callback_frequency) long.): A dictionary tracking the evolution of the log marginal likelihood at intervals defined by callback_frequency. The tensor can be accessed with history["log marginal likelihood"].

        Raises:
            NotFittedError: If the GaussianProcessRegressor model has not been fitted before training the kernel.
            TypeError: If the parameters are of wrong type.
            ValueError: If epochs is not a positive integer.
        """
        if not hasattr(self, "inverse_prior_covariance_matrix"):
            raise NotFittedError("GaussianProcessRegressor.fit() must be called before calculating the log marginal likelihood.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(optimiser, BaseOptimiser) and optimiser is not None:
            raise TypeError("optimiser must be from DLL.DeepLearning.Optimisers")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")

        optimiser = Adam() if optimiser is None else optimiser
        optimiser.initialise_parameters(list(self.covariance_function.parameters().values()))

        history = {"log marginal likelihood": torch.zeros(floor(epochs / callback_frequency))}

        for epoch in range(epochs):
            # form the derivative function
            def derivative(parameter_derivative):
                derivative = 0.5 * self.Y.T @ self.inverse_prior_covariance_matrix @ parameter_derivative @ self.inverse_prior_covariance_matrix @ self.Y

                if parameter_derivative.ndim == 2:
                    derivative = derivative.squeeze(1)
                    derivative -= 0.5 * torch.trace(self.inverse_prior_covariance_matrix @ parameter_derivative)
                else:
                    derivative = derivative.squeeze((1, 2))
                    derivative -= 0.5 * (self.inverse_prior_covariance_matrix @ parameter_derivative).diagonal(dim1=1, dim2=2).sum(dim=-1)
                # minus sign, since goal is to maximise the log_marginal_likelihood
                return -derivative

            # calculate the derivatives
            self.covariance_function.update(derivative, self.X)

            # update the parameters
            optimiser.update_parameters()

            self.prior_covariance_matrix = self._get_covariance_matrix(self.X, self.X) + (self.noise + self.epsilon) * torch.eye(len(self.X), device=self.device)
            self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)
            if epoch % callback_frequency == 0:
                lml = self.log_marginal_likelihood()
                history["log marginal likelihood"][int(epoch / callback_frequency)] = lml
                if verbose:
                    params = {}
                    for key, param in self.covariance_function.parameters().items():
                        if len(param) == 1:
                            params[key] = round(param.item(), 3)
                            continue
                        for i, param_ in enumerate(param):
                            params[key + "_" + str(i + 1)] = round(param_.item(), 3)
                    print(f"Epoch: {epoch + 1} - Log marginal likelihood: {lml} - Parameters: {params}")
        return history

    # def is_positive_definite(matrix):
    #     print("-----------------")
    #     if not torch.allclose(matrix, matrix.T):
    #         print(f"Matrix not symmetric: {torch.linalg.norm(matrix.T - matrix)}")
    #     eigenvalues = torch.linalg.eigvalsh(matrix)
    #     print(f"Minimum eigenvalue: {torch.min(eigenvalues).item()}")
