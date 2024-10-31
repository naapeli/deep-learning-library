import torch

from ....Exceptions import NotFittedError


class RidgeRegression:
    """
    Implements a linear regression model with L2 regularization.
    
    Args:
        alpha (int | float, optional): The regularization parameter. Larger alpha will force the l2 norm of the weights to be lower. Must be a positive real number. Defaults to 1.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        beta (torch.Tensor of shape (n_features + 1,)): The weights of the linear regression model. Available after fitting.
        residuals (torch.Tensor of shape (n_samples,)): The residuals of the fitted model. For a good fit, the residuals should be normally distributed with zero mean and constant variance. Available after fitting.
    """
    def __init__(self, alpha=1.0):
        if not isinstance(alpha, int | float) or alpha <= 0:
            raise ValueError("Alpha must be a positive integer.")
        
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fits the RidgeRegression model to the input data by minimizing the squared error.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
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
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")

        self.n_features = X.shape[1]
        X_with_const = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        identity = torch.eye(X_with_const.shape[1])
        identity[0, 0] = 0
        self.beta = torch.linalg.pinv(X_with_const.T @ X_with_const + self.alpha * identity) @ X_with_const.T @ y
        self.residuals = y - self.predict(X)

    def predict(self, X):
        """
        Applies the fitted RidgeRegression model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the RidgeRegression model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "beta"):
            raise NotFittedError("RidgeRegression.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        return X @ self.beta
