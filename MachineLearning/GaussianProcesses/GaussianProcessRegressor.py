import torch


class GaussianProcessRegressor:
    def __init__(self, covariance_function, noise=0):
        self.covariance_function = covariance_function
        self.noise = noise
        self._epsilon = 1e-10

    def _get_covariance_matrix(self, X1, X2):
        covariance = torch.tensor([[self.covariance_function(x1, x2) for x1 in X1] for x2 in X2]).T
        return covariance
        
    def fit(self, X, Y):
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        assert (len(Y.shape) == 2 and Y.shape[1] == 1) or len(Y.shape) == 1, "Y must be of shape (n_samples,)"
        assert X.shape[0] == Y.shape[0], "There must be an equal value of samples and values"
        self.X = X
        self.Y = Y
        self.inverse_prior_covariance_matrix = torch.linalg.inv(self._get_covariance_matrix(X, X) + (self.noise + self._epsilon) * torch.eye(len(X)))

    def predict(self, X):
        assert hasattr(self, "inverse_prior_covariance_matrix"), "GaussianProcessRegressor.fit(x, y) must be called before predicting"
        assert X.shape[1] == self.X.shape[1], "The input dimension must match the fitted dimension"
        k_1 = self._get_covariance_matrix(self.X, X)
        k_2 = self._get_covariance_matrix(X, X) + self._epsilon * torch.eye(len(X))
        mean = k_1.T @ self.inverse_prior_covariance_matrix @ self.Y
        posterior_covariance = k_2 - k_1.T @ self.inverse_prior_covariance_matrix @ k_1
        # is_positive_definite(self._get_covariance_matrix(self.X, self.X))
        # is_positive_definite(k_2)
        # is_positive_definite(self.inverse_prior_covariance_matrix)
        # is_positive_definite(posterior_covariance)
        return mean, posterior_covariance

def is_positive_definite(matrix):
    print("-----------------")
    if not torch.allclose(matrix, matrix.T):
        print(f"Matrix not symmetric: {torch.linalg.norm(matrix.T - matrix)}")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    print(f"Minimum eigenvalue: {torch.min(eigenvalues).item()}")
