import torch


class GaussianProcessRegressor:
    def __init__(self, covariance_function, noise=0, epsilon=1e-10):
        self.covariance_function = covariance_function
        self.noise = noise
        self.epsilon = epsilon

    def _get_covariance_matrix(self, X1, X2):
        covariance = torch.tensor([[self.covariance_function(x1, x2) for x1 in X1] for x2 in X2]).T
        return covariance
        
    def fit(self, X, Y):
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        assert (len(Y.shape) == 2 and Y.shape[1] == 1) or len(Y.shape) == 1, "Y must be of shape (n_samples,)"
        assert X.shape[0] == Y.shape[0], "There must be an equal value of samples and values"
        self.X = X
        self.Y = Y
        self.prior_covariance_matrix = self._get_covariance_matrix(X, X) + (self.noise + self.epsilon) * torch.eye(len(X))
        self.inverse_prior_covariance_matrix = torch.linalg.inv(self.prior_covariance_matrix)

    def predict(self, X):
        assert hasattr(self, "inverse_prior_covariance_matrix"), "GaussianProcessRegressor.fit(x, y) must be called before predicting"
        assert X.shape[1] == self.X.shape[1], "The input dimension must match the fitted dimension"
        k_1 = self._get_covariance_matrix(self.X, X)
        k_2 = self._get_covariance_matrix(X, X) + self.epsilon * torch.eye(len(X))
        mean = k_1.T @ self.inverse_prior_covariance_matrix @ self.Y
        posterior_covariance = k_2 - k_1.T @ self.inverse_prior_covariance_matrix @ k_1
        # is_positive_definite(self._get_covariance_matrix(self.X, self.X))
        # is_positive_definite(k_2)
        # is_positive_definite(self.inverse_prior_covariance_matrix)
        # is_positive_definite(posterior_covariance)
        return mean, posterior_covariance
    
    def log_marginal_likelihood(self):
        assert hasattr(self, "inverse_prior_covariance_matrix"), "GaussianProcessRegressor.fit(x, y) must be called before getting the log-marginal-likelihood"
        L = torch.linalg.cholesky(self.prior_covariance_matrix)
        alpha = torch.cholesky_solve(self.Y, L)
        lml = -0.5 * self.Y.T @ alpha - torch.sum(torch.log(torch.diagonal(L))) - 0.5 * len(self.Y) * torch.log(torch.tensor(2 * torch.pi))
        # lml = -0.5 * (self.Y - 0).T @ self.inverse_prior_covariance_matrix @ (self.Y - 0) - 0.5 * torch.log(torch.linalg.det(self.prior_covariance_matrix)) - self.X.shape[0] / 2 * torch.log(torch.tensor(2 * torch.pi))
        return lml.item()
    
    def log_marginal_likelihood_derivative(self, parameter):
        pass


# def is_positive_definite(matrix):
#     print("-----------------")
#     if not torch.allclose(matrix, matrix.T):
#         print(f"Matrix not symmetric: {torch.linalg.norm(matrix.T - matrix)}")
#     eigenvalues = torch.linalg.eigvalsh(matrix)
#     print(f"Minimum eigenvalue: {torch.min(eigenvalues).item()}")
