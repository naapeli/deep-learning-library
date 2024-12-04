import torch

from ....Exceptions import NotFittedError


class LinearRegression:
    """
    Implements the basic linear regression model.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        beta (torch.Tensor of shape (n_features + 1,)): The weights of the linear regression model. Available after fitting.
        residuals (torch.Tensor of shape (n_samples,)): The residuals of the fitted model. For a good fit, the residuals should be normally distributed with zero mean and constant variance. Available after fitting.
    """
    def fit(self, X, y, include_bias=True):
        """
        Fits the LinearRegression model to the input data by minimizing the squared error.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            include_bias (bool, optional): Decides if a bias is included in a model. Defaults to True.
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
        if not isinstance(include_bias, bool):
            raise TypeError("include_bias must be a boolean.")

        self.include_bias = include_bias
        self.n_features = X.shape[1]
        X_ = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1) if self.include_bias else X
        self.beta = torch.linalg.lstsq(X_.T @ X_, X_.T @ y).solution
        self.residuals = y - self.predict(X)

    def predict(self, X):
        """
        Applies the fitted LinearRegression model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the LinearRegression model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "beta"):
            raise NotFittedError("LinearRegression.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        if self.include_bias: X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        return X @ self.beta    
