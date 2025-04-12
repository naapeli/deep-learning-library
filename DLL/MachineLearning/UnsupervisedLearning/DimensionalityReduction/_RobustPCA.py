import torch
from math import sqrt, floor

from ....Exceptions import NotFittedError
from . import PCA


class RobustPCA:
    """
    Robust version of Principal Component Analysis (PCA).

    Args:
        n_components (int): Number of principal components to keep. The number must be a positive integer.
        method (str, optional): The method used for the robust estimator. Must be one of "mcd" or "decomposition". Defaults to "mcd".
    Attributes:
        components (torch.Tensor): Principal components extracted from the data.
        explained_variance (torch.Tensor): Variance explained by the selected components.
    """
    def __init__(self, n_components=2, method="mcd", epsilon=1e-10):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if method not in ["mcd", "decomposition"]:
            raise ValueError('method must be one of "mcd" or "decomposition".')
        
        self.n_components = n_components
        self.method = method
        self.epsilon = epsilon
        self.pca = PCA(n_components, epsilon)

    def fit(self, X, epochs=1000, normalize=True, proportion=0.5, n_tries=20):
        """
        Fits the RobustPCA model to the input data by calculating the principal components. If method is "decomposition", finds matricies L and S such that X = L + S, where L is low rank and S is sparse. Then applies PCA to L. If method is mcd, tries to find a subset of the data that has the minimum covariance determinant and uses that as an estimate of the covariance.
    
        The input data is always centered and if `normalize=True`, also normalized so that the standard deviation is 1 along each axis.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            epochs (int, optional): Determines how many iterations are used for finding the L and S matricies. Defaults to 1000. Must be a positive integer.
            normalize (bool, optional): Whether to normalize the data before computing the PCA. Defaults to True.
            proportion (float, optional): The proportion of data, which is used to approximate the covariance matrix. Is ignored unless method is "mcd". Must be in range (0, 1). Defaults to 0.5.
            n_tries (int, optional): The amount of attempts of randomly selecting the samples for the covariance estimate. Must be a positive integer. Defaults to 20.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be positive integer.")
        if not isinstance(normalize, bool):
            raise TypeError("The normalize parameter must be a boolean.")
        if X.ndim != 2 or X.shape[0] == 1:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast 2 samples.")
        if not isinstance(proportion, float) or proportion <= 0 or proportion >= 1:
            raise ValueError("proportion must be in range (0, 1).")
        if not isinstance(n_tries, int) or n_tries <= 0:
            raise ValueError("n_tries must be a positive integer.")

        self.normalize = normalize
        if self.method == "decomposition":
            n1, n2 = X.shape
            mu = n1 * n2 / (4 * torch.sum(torch.abs(X)))
            l = 1 / sqrt(max(n1, n2))
            thresh = 1e-7 * torch.linalg.norm(X, ord="fro")
            L = torch.zeros_like(X)
            S = torch.zeros_like(X)
            Y = torch.zeros_like(X)
            i = 0
            while torch.linalg.norm(X - L - S, ord="fro") > thresh and i < epochs:
                L = self._SVT(X - S + (1 / mu) * Y, 1 / mu)
                S = self._shrink(X - L + (1 / mu) * Y, l / mu)
                Y = Y + mu * (X - L - S)
                i = i + 1
            self.pca.fit(L, normalize, False)
        elif self.method == "mcd":
            covariance, self.mean, self.std = self._min_cov_det(X, proportion, normalize, n_tries)
            self.pca.fit(covariance, normalize, True)

    def _shrink(self, X, tau):
        return torch.sign(X) * torch.max(torch.abs(X) - tau, torch.zeros_like(X))

    def _SVT(self, X, tau):
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return U @ self._shrink(torch.diag(S), tau) @ Vh
    
    def _min_cov_det(self, X, proportion, normalize, n_tries):
        N = floor(proportion * len(X))
        best_covariance_matrix = torch.cov(X.T)
        lowest_determinant = torch.linalg.det(best_covariance_matrix)
        best_mean = X.mean(dim=0)
        best_std = X.std(dim=0, unbiased=True) if normalize else None
        for _ in range(n_tries):
            indices = torch.randperm(len(X))[:N]
            mean = X[indices].mean(dim=0)
            X_centered = X[indices] - mean
            if normalize:
                std = X_centered.std(dim=0, unbiased=True)
                X_centered = X_centered / (std + self.epsilon)
            covariance = torch.cov(X_centered.T)
            determinant = torch.linalg.det(covariance)
            if determinant < lowest_determinant:
                best_covariance_matrix = covariance
                lowest_determinant = determinant
                best_mean = mean
                if normalize: best_std = std
        return best_covariance_matrix, best_mean, best_std
    
    def transform(self, X):
        """
        Applies the fitted RobustPCA model to the input data, transforming it into the reduced feature space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the principal component space.
        Raises:
            NotFittedError: If the RobustPCA model has not been fitted before transforming.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self.pca, "from_covariance"):
            raise NotFittedError("RobustPCA.fit() must be called before transforming.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.pca.components.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        if self.method == "mcd":
            X = X - self.mean
            if self.normalize: X = X / (self.std + self.epsilon)
        return self.pca.transform(X)
    
    def fit_transform(self, X, epochs=1000, normalize=True):
        """
        First finds the principal components of X and then transforms X to fitted space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
            epochs (int, optional): Determines how many iterations are used for finding the L and S matricies. Defaults to 1000. Must be a positive integer.
            normalize (bool, optional): Whether to normalize the data before computing the PCA. Defaults to True.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the principal component space.
        """
        
        self.fit(X, epochs=epochs, normalize=normalize)
        return self.transform(X)
