import torch
from ....DeepLearning.Optimisers.ADAM import Adam


class GaussianProcessRegressor:
    def __init__(self, covariance_function, noise=0, epsilon=1e-5, device=torch.device("cpu")):
        self.covariance_function = covariance_function
        self.noise = noise
        self.epsilon = epsilon
        self.device = device

    def _get_covariance_matrix(self, X1, X2):
        return self.covariance_function(X1, X2).to(X1.dtype).to(self.device)

    def fit(self, X, Y):
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        assert (len(Y.shape) == 2 and Y.shape[1] == 1) or len(Y.shape) == 1, "Y must be of shape (n_samples,)"
        assert X.shape[0] == Y.shape[0], "There must be an equal value of samples and values"
        variance = torch.var(Y, dim=0)
        mean = torch.mean(Y, dim=0)
        assert torch.allclose(variance, torch.ones_like(variance)), "Remember to normalise Y to variance 1"
        assert torch.allclose(mean, torch.zeros_like(mean)), "Remember to normalise Y to mean 0"
        self.X = X
        self.Y = Y
        self.prior_covariance_matrix = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X), device=self.device)
        self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)

    def predict(self, X):
        assert hasattr(self, "inverse_prior_covariance_matrix"), "GaussianProcessRegressor.fit(x, y) must be called before predicting"
        assert X.shape[1] == self.X.shape[1], "The input dimension must match the fitted dimension"
        k_1 = self._get_covariance_matrix(self.X, X)
        k_2 = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X), device=self.device)
        mean = k_1.T @ self.inverse_prior_covariance_matrix @ self.Y
        posterior_covariance = k_2 - k_1.T @ self.inverse_prior_covariance_matrix @ k_1
        return mean, posterior_covariance
    
    def log_marginal_likelihood(self):
        assert hasattr(self, "inverse_prior_covariance_matrix"), "GaussianProcessRegressor.fit(x, y) must be called before getting the log-marginal-likelihood"
        L = torch.linalg.cholesky(self.prior_covariance_matrix)
        alpha = torch.cholesky_solve(self.Y, L)
        lml = -0.5 * self.Y.T @ alpha - torch.sum(torch.log(torch.diagonal(L))) - 0.5 * len(self.Y) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        return lml.item()
    
    def train_kernel(self, epochs=10, optimiser=Adam()):
        assert hasattr(self, "X"), "GaussianProcessRegressor.fit(x, y) must be called before training"
        optimiser.initialise_parameters(self.covariance_function.parameters())
        for epoch in range(1, epochs + 1):
            # form the derivative function
            def derivative(parameter_derivative):
                derivative = 0.5 * self.Y.T @ self.inverse_prior_covariance_matrix @ parameter_derivative @ self.inverse_prior_covariance_matrix @ self.Y
                derivative -= 0.5 * torch.trace(self.inverse_prior_covariance_matrix @ parameter_derivative)
                # minus sign, since goal is to maximise the log_marginal_likelihood
                return -derivative

            # calculate the derivatives
            self.covariance_function.update(derivative, self.X, noise=self.noise, epsilon=self.epsilon)

            # update the parameters
            optimiser.update_parameters()

            self.prior_covariance_matrix = self._get_covariance_matrix(self.X, self.X) + (self.noise + self.epsilon) * torch.eye(len(self.X), device=self.device)
            self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)
            print(f"Epoch: {epoch} - Log-marginal-likelihood: {self.log_marginal_likelihood()}")
        
    # def is_positive_definite(matrix):
    #     print("-----------------")
    #     if not torch.allclose(matrix, matrix.T):
    #         print(f"Matrix not symmetric: {torch.linalg.norm(matrix.T - matrix)}")
    #     eigenvalues = torch.linalg.eigvalsh(matrix)
    #     print(f"Minimum eigenvalue: {torch.min(eigenvalues).item()}")
