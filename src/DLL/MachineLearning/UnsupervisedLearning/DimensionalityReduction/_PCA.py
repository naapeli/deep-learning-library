import torch

from ....Exceptions import NotFittedError


class PCA:
    """
    Principal Component Analysis (PCA) class for dimensionality reduction.

    Args:
        n_components (int): Number of principal components to keep. The number must be a positive integer.
    Attributes:
        components (torch.Tensor): Principal components extracted from the data.
        explained_variance (torch.Tensor): Variance explained by the selected components.
    """
    def __init__(self, n_components=2, epsilon=1e-10):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = n_components
        self.epsilon = epsilon

    def fit(self, X, normalize=True):
        """
        Fits the PCA model to the input data by calculating the principal components.
        
        The input data is always centered and if `normalize=True`, also normalized so that the standard deviation is 1 along each axis.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            normalize (bool, optional): Whether to normalize the data before computing the PCA. Defaults to True.
        Returns:
            None
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor or if the `normalize` parameter is not boolean.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if not isinstance(normalize, bool):
            raise TypeError("The normalize parameter must be a boolean.")
        if X.ndim != 2 or X.shape[0] == 1:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast 2 samples.")

        self.normalize = normalize
        self.mean = X.mean(dim=0)
        X = (X - self.mean)
        if self.normalize:
            self.standard_deviation = X.std(dim=0, unbiased=True)
            X = X / (self.standard_deviation + self.epsilon)
        covariance = torch.cov(X.T)
        eig_vals, eig_vecs = torch.linalg.eig(covariance)
        indicies = torch.argsort(eig_vals.real, descending=True)
        self.components = eig_vecs.real.T[indicies][:self.n_components]
        self.explained_variance = eig_vals.real[indicies][:self.n_components]
    
    def transform(self, X):
        """
        Applies the fitted PCA model to the input data, transforming it into the reduced feature space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the principal component space.
        Raises:
            NotFittedError: If the PCA model has not been fitted before transforming.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "mean"):
            raise NotFittedError("PCA.fit() must be called before transforming.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != len(self.mean):
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        X = (X - self.mean)
        if self.normalize:
            X = X / (self.standard_deviation + self.epsilon)
        return X @ self.components.T
    
    def fit_transform(self, X, normalize=True):
        """
        First finds the principal components of X and then transforms X to fitted space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
            normalize (bool, optional): Whether to normalize the data before computing the PCA. Defaults to True.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the principal component space.
        """
        
        self.fit(X, normalize=normalize)
        return self.transform(X)
