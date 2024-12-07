import torch

from ....Exceptions import NotFittedError


class LDA:
    """
    Linear discriminant analysis (LDA) class for dimensionality reduction.

    Args:
        n_components (int): Number of principal components to keep. The number must be a positive integer.
    Attributes:
        components (torch.Tensor): Components extracted from the data.
        n_features (int): The number of features in the input.
    """
    def __init__(self, n_components=2):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        
        self.n_components = n_components
    
    def fit(self, X, y):
        """
        Fits the LDA model to the input data by calculating the components.
        
        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] == 1:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast 2 samples.")
        
        classes = torch.unique(y)
        self.n_features = X.shape[1]
        X_mean = torch.mean(X, dim=0)

        Sw = torch.zeros((self.n_features, self.n_features), dtype=X.dtype)
        Sb = torch.zeros((self.n_features, self.n_features), dtype=X.dtype)

        for current_class in classes:
            X_c = X[y == current_class]

            C_mean = torch.mean(X_c, dim=0)
            Sw += (X_c - C_mean).T @ (X_c - C_mean)
            mean_diff = (C_mean - X_mean)
            Sb += len(X_c) * (mean_diff @ mean_diff)
        
        A = torch.linalg.lstsq(Sw, Sb).solution
        eig_vals, eig_vecs = torch.linalg.eig(A)
        indicies = torch.argsort(eig_vals.real, descending=True)
        self.components = eig_vecs.real.T[indicies][:self.n_components].T

    def transform(self, X):
        """
        Applies the fitted LDA model to the input data, transforming it into the reduced feature space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the component space.
        Raises:
            NotFittedError: If the LDA model has not been fitted before transforming.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "components"):
            raise NotFittedError("LDA.fit() must be called before transforming.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        return X @ self.components
    
    def fit_transform(self, X, y):
        """
        First finds the components of X and then transforms X to fitted space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
        Returns:
            X_new (torch.Tensor of shape (n_samples, n_components)): The data transformed into the component space.
        """
        
        self.fit(X, y)
        return self.transform(X)
