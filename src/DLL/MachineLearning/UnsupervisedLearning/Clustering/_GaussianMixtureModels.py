import torch

from ....Exceptions import NotFittedError


class GaussianMixture:
    """
    Gaussian mixtre model. Fits k gaussian distributions onto the data using maximum likelihood estimation.

    Args:
        k, (int, optional): The number of gaussian distributions (clusters). Must be a positive integer. Defaults to 3.
        max_iters (int, optional): The maximum number of iterations. Must be a positive integer. Defaults to 10.
    """
    def __init__(self, k=3, max_iters=10, tol=1e-5):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError("max_iters must be a positive integer.")

        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def _pdf(self, X):
        # X (n_samples, n_features)
        X_minus_mu = X[:, None, :] - self.mus
        exponent = torch.einsum('nkd,kde,nke->nk', X_minus_mu, torch.linalg.inv(self.sigmas), X_minus_mu)
        const = (2 * torch.pi) ** (-0.5 * self.n_features) * torch.linalg.det(self.sigmas) ** (-0.5)
        return const * torch.exp(-0.5 * exponent)

    def _expectation_step(self, X):
        # Compute the posterior given prior, mean and covariance.
        likelihood = self._pdf(X)
        posterior = likelihood * self.prior[None, :]
        norm_const = torch.sum(posterior, axis=1)
        normalised_posterior = posterior / norm_const[:, None]
        return normalised_posterior

    def _maximization_step(self, X, posterior):
        posterior_sum = torch.sum(posterior, axis=0)
        new_prior = posterior_sum / len(X)

        posterior_mus = (X.T @ posterior / posterior_sum).T

        X_centered = X[:, None, :] - posterior_mus
        X_weighted = X_centered * posterior[:, :, None]
        posterior_sigmas = torch.einsum('nkd,nke->kde', X_weighted, X_centered) / posterior_sum[:, None, None]
        return posterior_mus, posterior_sigmas, new_prior
    
    def _log_likelihood(self, X):
        return torch.log(torch.sum(self._pdf(X) * self.prior))

    def fit(self, X, verbose=False):
        """
        Fits the k gaussian distributions to the data using maximum likelihood estimation.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be clustered.
            verbose (bool, optional): Determines if the likelihood should be calculated and printed during training. Must be a boolean. Defaults to False.
        """
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            raise TypeError("X must be a 2 dimensional torch tensor.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")

        self.n_features = X.shape[1]
        # self.mus = torch.randn(self.k, n_features, dtype=X.dtype, device=X.device)
        random_indices = torch.randperm(len(X))[:self.k]  # Random permutation of indices, pick first k
        self.mus = X[random_indices].clone()
        self.sigmas = torch.stack([torch.eye(self.n_features, dtype=X.dtype, device=X.device) for _ in range(self.k)])
        self.prior = torch.ones(self.k, dtype=X.dtype, device=X.device)/self.k
        if verbose: print(f"Initial log-likelihood: {self._log_likelihood(X)}")

        for i in range(self.max_iters):
            posterior = self._expectation_step(X)
            mus, sigmas, prior = self._maximization_step(X, posterior)
            if torch.all(mus - self.mus < self.tol) and torch.all(sigmas - self.sigmas < self.tol):
                break
            self.taus, self.mus, self.sigmas = prior, mus, sigmas
            if verbose: print(f"Epoch: {i + 1} - Log-likelihood: {self._log_likelihood(X)}")

    def predict(self, X):
        """
        Predicts the clusters of the data according to the fitted distributions.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be clustered.

        Returns:
            torch.Tensor of shape (n_samples,): A tensor of labels corresponding to classes.
        """
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            raise TypeError("X must be a 2 dimensional torch tensor.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of features as the training data.")
        if not hasattr(self, "mus"):
            raise NotFittedError("GaussianMixture.fit() must be fitted before predicting.")


        posterior = self._expectation_step(X)
        return torch.argmax(posterior, dim=1)

    def predict_proba(self, X):
        """
        Predicts the probabilities of the data being in the fitted distributions.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be clustered.

        Returns:
            torch.Tensor of shape (n_samples, k): A tensor of probabilities of the data being in the fitted distributions.
        """
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            raise TypeError("X must be a 2 dimensional torch tensor.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of features as the training data.")
        if not hasattr(self, "mus"):
            raise NotFittedError("GaussianMixture.fit() must be fitted before predicting.")

        posterior = self._expectation_step(X)
        return posterior
