import torch
from math import floor

from ..Kernels import _Base
from ....DeepLearning.Optimisers import ADAM
from ....DeepLearning.Optimisers._BaseOptimiser import BaseOptimiser
from ....Exceptions import NotFittedError


class GaussianProcessClassifier:
    """
    Implements the Gaussian process classification model for binary classes. This model can be extended to multiclass classification by OvO or OvR.

    Args:
        covariance_function (:ref:`kernel_section_label`, optional): The kernel function expressing how similar are different samples.
        noise (int | float, optional): The artificially added noise to the model. Is added as variance to each sample. Must be non-negative. Defaults to 0.
        epsilon (float, optional): Implemented similarly to noise. Makes sure the covariance matrix is positive definite and hence invertible. Must be positive. Defaults to 1e-5. If one gets a RunTimeError for a matrix not being invertible, one should increase this parameter.
        device (torch.device, optional): The device of all matrices. Defaults to torch.device("cpu").
        
    Attributes:
        n_features (int): The number of features. Available after fitting.
    """
    def __init__(self, covariance_function, noise=0, n_iter_laplace_mode=100, epsilon=1e-5, device=torch.device("cpu")):
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
        self.n_iter_laplace_mode = n_iter_laplace_mode
        self.epsilon = epsilon
        self.device = device

    def _get_covariance_matrix(self, X1, X2):
        return self.covariance_function(X1, X2).to(X1.dtype).to(self.device)

    def fit(self, X, y):
        """
        Fits the GaussianProcessClassifier model to the input data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample. Should be normalized to zero mean and one variance. Every element must be in [0, 1].
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
        vals = torch.unique(y).numpy()
        if set(vals) != {0, 1}:
            raise ValueError("y must only contain the values in [0, 1].")
        # variance = torch.var(y)
        # mean = torch.mean(y)
        # if not torch.allclose(variance, torch.ones_like(variance)):
        #     raise ValueError("y must have one variance. Use DLL.Data.Preprocessing.StandardScaler to scale the data.")
        # if not torch.allclose(mean, torch.zeros_like(mean)):
        #     raise ValueError("y must have zero mean. Use DLL.Data.Preprocessing.StandardScaler to scale the data.")

        self.n_features = X.shape[1]
        self.X = X
        self.Y = y
        self.prior_covariance_matrix = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X), device=self.device)
        self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)
        _, (self._pi, self._W_sr, self._L, _, _) = self._posterior_mode()  # save necessary things for the predictions

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

        K = self._get_covariance_matrix(self.X, X)
        f_star = K.T @ (self.Y - self._pi)
        return torch.where(f_star > 0, 1, 0)
    
    def predict_proba(self, X):
        """
        Applies the fitted GaussianProcessRegressor model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            probabilities (tuple[torch.Tensor of shape (n_samples,)): The probabilities that the class belongs to class 1.
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

        K = self._get_covariance_matrix(self.X, X)
        f_star = K.T @ (self.Y - self._pi)
        v = torch.linalg.solve(self._L, self._W_sr.unsqueeze(-1) * K)
        var_f_star = torch.diag(self._get_covariance_matrix(X, X)) - torch.diag(v.T @ v)

        LAMBDAS = torch.tensor([0.41, 0.4, 0.37, 0.44, 0.39]).unsqueeze(-1)
        COEFS = torch.tensor([-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]).unsqueeze(-1)
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = torch.sqrt(torch.pi / alpha) * torch.erf(gamma * torch.sqrt(alpha / (alpha + LAMBDAS ** 2))) / (2 * torch.sqrt(var_f_star * 2 * torch.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()
        # return torch.vstack((1 - pi_star, pi_star)).T
        return pi_star
    
    def log_marginal_likelihood(self):
        """
        Computes the log marginal likelihood of the current model. This value is used to optimize hyperparameters.

        Returns:
            log marginal likelihood (float): The log marginal likelihood of the current model.
        """
        if not hasattr(self, "inverse_prior_covariance_matrix"):
            raise NotFittedError("GaussianProcessRegressor.fit() must be called before calculating the log marginal likelihood.")

        return self._posterior_mode()[0]
    
    def _posterior_mode(self):
        # adapted from sklearn source code: https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/gaussian_process/_gpc.py#L269
        f = self._f_cached if hasattr(self, "_f_cached") else torch.zeros_like(self.Y)
        log_marginal_likelihood = -torch.inf
        K = self.prior_covariance_matrix
        for _ in range(self.n_iter_laplace_mode):
            pi = 1 / (1 + torch.exp(-f))
            W = pi * (1 - pi)
            W_sr = torch.sqrt(W)
            W_sr_K = W_sr.unsqueeze(-1) * K
            B = torch.eye(len(W)) + W_sr_K * W_sr
            L = torch.linalg.cholesky(B)
            b = W * f + (self.Y - pi)
            a = b - W_sr * torch.cholesky_solve((W_sr_K @ b).unsqueeze(-1), L).squeeze(-1)
            f = K @ a
            lml = -0.5 * a @ f - torch.log1p(torch.exp(-(self.Y * 2 - 1) * f)).sum() - torch.log(torch.diag(L)).sum()
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml
        self._f_cached = f
        return log_marginal_likelihood, (pi, W_sr, L, b, a)
    
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

        optimiser = ADAM() if optimiser is None else optimiser
        optimiser.initialise_parameters(list(self.covariance_function.parameters().values()))

        history = {"log marginal likelihood": torch.zeros(floor(epochs / callback_frequency))}

        for epoch in range(epochs):
            # form the derivative function
            def derivative(parameter_derivative):
                K = self.prior_covariance_matrix
                _, (pi, W_sr, L, b, a) = self._posterior_mode()
                R = W_sr.unsqueeze(-1) * torch.cholesky_solve(torch.diag(W_sr), L)
                C = torch.linalg.solve(L, W_sr.unsqueeze(-1) * K)
                s_2 = -0.5 * (torch.diag(K) - torch.diag(C.T @ C)) * (pi * (1 - pi) * (1 - 2 * pi))
                C = parameter_derivative
                s_1 = 0.5 * (a @ C) @ a - 0.5 * R.T.ravel() @ C.ravel()
                b = C @ (self.Y - pi)
                s_3 = b - K @ (R @ b)
                derivative = s_1 + s_2 @ s_3
                # minus sign, since goal is to maximise the log_marginal_likelihood
                return -derivative.unsqueeze(-1)

            # calculate the derivatives
            self.covariance_function.update(derivative, self.X)

            # update the parameters
            optimiser.update_parameters()

            self.prior_covariance_matrix = self._get_covariance_matrix(self.X, self.X) + (self.noise + self.epsilon) * torch.eye(len(self.X), device=self.device)
            self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)
            if epoch % callback_frequency == 0:
                lml = self.log_marginal_likelihood().item()
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
        
        _, (self._pi, self._W_sr, self._L, _, _) = self._posterior_mode()  # save necessary things for the predictions
        return history

    # def is_positive_definite(matrix):
    #     print("-----------------")
    #     if not torch.allclose(matrix, matrix.T):
    #         print(f"Matrix not symmetric: {torch.linalg.norm(matrix.T - matrix)}")
    #     eigenvalues = torch.linalg.eigvalsh(matrix)
    #     print(f"Minimum eigenvalue: {torch.min(eigenvalues).item()}")
