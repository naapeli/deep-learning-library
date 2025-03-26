import torch

from ....Exceptions import NotFittedError


class QDA:
    """
    Quadratic discriminant analysis (LDA) class for classification.

    Attributes:
        n_features (int): The number of features in the input.
    """

    def fit(self, X, y):
        """
        Fits the QDA model to the input data by calculating the class means and covariances.
        
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
        
        self.classes = torch.unique(y)
        self.n_features = X.shape[1]

        self.class_means = []
        self.class_covariances = []

        for current_class in self.classes:
            X_c = X[y == current_class]
            self.class_means.append(torch.mean(X_c, dim=0))
            self.class_covariances.append(torch.cov(X_c.T, correction=0))

    def predict(self, X):
        """
        Applies the fitted QDA model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be transformed.
        Returns:
            y (torch.Tensor of shape (n_samples,)): The predicted labels.
        Raises:
            NotFittedError: If the QDA model has not been fitted before transforming.
        """
        if not hasattr(self, "class_means"):
            raise NotFittedError("QDA.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] == 1:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast 2 samples.")

        min_dists = torch.full((len(X),), torch.inf, dtype=X.dtype)
        y = torch.zeros_like(min_dists, dtype=X.dtype)
        
        for current_class, class_mean, class_covariance in zip(self.classes, self.class_means, self.class_covariances):
            diff = X - class_mean
            # mahalanobis = torch.sum(diff @ torch.linalg.lstsq(class_covariance, diff.T).solution, dim=1)
            # mahalanobis = torch.sum(diff @ torch.linalg.inv(class_covariance) * diff, dim=1)
            mahalanobis = torch.sum(torch.linalg.lstsq(class_covariance, diff.T).solution.T * diff, dim=1)
            mask = mahalanobis < min_dists
            y[mask] = current_class
            min_dists[mask] = mahalanobis[mask]
        return y
