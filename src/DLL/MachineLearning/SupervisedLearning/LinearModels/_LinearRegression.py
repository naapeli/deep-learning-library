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
    def fit(self, X, y, include_bias=True, method="ols", sample_weight=None):
        """
        Fits the LinearRegression model to the input data by minimizing the squared error.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            include_bias (bool, optional): Decides if a bias is included in a model. Defaults to True.
            method (str, optional): Determines if the loss function is ordinary least squares or total least squares. Must be one of "ols" or "tls". Defaults to "ols".
            sample_weight (torch.Tensor of shape (n_samples,) or None): A weight given to each sample in the regression. If None, this parameter is ignored. sample_weight is ignored if method == "tls".
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
        if method not in ["ols", "tls"]:
            raise ValueError('method must be one of "ols" or "tls".')
        if not isinstance(sample_weight, torch.Tensor) and sample_weight is not None:
            raise TypeError("sample_weight must be torch.Tensor or None.")
        if isinstance(sample_weight, torch.Tensor) and (sample_weight.ndim != 1 or len(X) != len(sample_weight)):
            raise ValueError("sample_weight must be of shape (n_samples,)")

        self.include_bias = include_bias
        self.n_features = X.shape[1]
        X_ = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1) if self.include_bias else X
        if method == "ols":
            sample_weight = torch.diag(sample_weight) if sample_weight is not None else torch.eye(len(X))
            self.beta = torch.linalg.lstsq(X_.T @ sample_weight @ X_, X_.T @ sample_weight @ y).solution
        elif method == "tls":
            C = torch.cat((X_, y.unsqueeze(1)), dim=1)            
            _, _, Vt = torch.linalg.svd(C, full_matrices=False)
            v_min = Vt[-1]
            self.beta = -v_min[:-1] / v_min[-1]

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
