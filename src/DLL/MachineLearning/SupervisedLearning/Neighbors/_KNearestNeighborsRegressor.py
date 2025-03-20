import torch
from functools import partial

from ....Exceptions import NotFittedError


class KNNRegressor:
    """
    The k-nearest neighbors regressor model. Looks for the k closest samples with respect to a distance metric and calculates the average according to some weight function.
    
    Args:
        k (int, optional): The number of closest samples considered for the predictions. Must be a positive integer. Defaults to 3.
        metric (str, optional): A distance metric for the closest points. Must be one of "euclidian" or "manhattan". Defaults to "euclidian".
        weight (str, optional): A weight function that decides how important are the nearest k samples. Must be in ["uniform", "distance", "gaussian"]. Defaults to "gaussian".
    """
    def __init__(self, k=3, metric="euclidian", weight="gaussian"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if metric not in ["euclidian", "manhattan"]:
            raise ValueError('metric must be on of "euclidian" or "manhattan".')
        if weight not in ["uniform", "distance", "gaussian"]:
            raise ValueError('weight must be in ["uniform", "distance", "gaussian"].')

        self.k = k
        self.metric = partial(self._metric, metric)
        self.weight = partial(self._weight, weight)
    
    def _metric(self, metric, X1, X2):
        if metric == "euclidian":
            return ((X1 - X2) ** 2).sum(dim=2).sqrt()
        elif metric == "manhattan":
            return torch.abs(X1 - X2).sum(dim=2)
    
    def _weight(self, weight, distances):
        if weight == "uniform":
            return torch.ones_like(distances)
        elif weight == "distance":
            return 1 / (distances + 1e-10)
        elif weight == "gaussian":
            return torch.exp(-distances ** 2)

    def fit(self, X, y):
        """
        Fits the KNNRegressor model to the input data by storing the input and target matricies.

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
            raise TypeError("The input matrix and the target matrix must be PyTorch tensors.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The target values must be 1 dimensional with the same number of samples as the input data")

        self.X = X
        self.y = y

    def predict(self, X):
        """
        Applies the fitted KNNRegressor model to the input data, predicting the target values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            targets (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the KNNRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "X"):
            raise NotFittedError("KNNRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        distances: torch.Tensor = self.metric(self.X.unsqueeze(0), X.unsqueeze(1)) # (len(X), len(self.X))
        indicies = distances.topk(self.k, largest=False).indices
        k_values = self.y[indicies]
        k_distances = distances.gather(1, indicies)
        k_weights = self.weight(k_distances)
        return (k_values * k_weights).sum(dim=1) / (k_weights.sum(dim=1) + 1e-10)
