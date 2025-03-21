import torch
from functools import partial

from ....Exceptions import NotFittedError


class KNNClassifier:
    """
    The k-nearest neighbors classifier model. Looks for the k closest samples with respect to a distance metric and calculates the most common label.
    
    Args:
        k (int, optional): The number of closest samples considered for the predictions. Must be a positive integer. Defaults to 3.
        metric (str, optional): A distance metric for the closest points. Must be one of "euclidian" or "manhattan". Defaults to "euclidian".
    """
    def __init__(self, k=3, metric="euclidian"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if metric not in ["euclidian", "manhattan"]:
            raise ValueError('metric must be on of "euclidian" or "manhattan".')

        self.k = k
        self.metric = partial(self._metric, metric)
    
    def _metric(self, metric, X1, X2):
        if metric == "euclidian":
            return ((X1 - X2) ** 2).sum(dim=2).sqrt()
        elif metric == "manhattan":
            return torch.abs(X1 - X2).sum(dim=2)

    def fit(self, X, y):
        """
        Fits the KNNClassifier model to the input data by storing the input and label matricies.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature. There must be atleast k samples.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the label matrix is not a PyTorch tensor.
            ValueError: If the input matrix or the label matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the label matrix must be PyTorch tensors.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        if X.shape[0] < self.k:
            raise ValueError("There must be atleast k samples.")
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        self.X = X
        self.y = y

    def predict(self, X):
        """
        Applies the fitted KNNClassifier model to the input data, predicting the labels.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the KNNClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "X"):
            raise NotFittedError("KNNClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        distances: torch.Tensor = self.metric(self.X.unsqueeze(0), X.unsqueeze(1)) # (len(X), len(self.X))
        indicies = distances.topk(self.k, largest=False).indices
        k_labels = self.y[indicies]
        return torch.mode(k_labels, dim=1).values
    
    def predict_proba(self, X):
        """
        Applies the fitted KNNClassifier model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,) for binary and (n_samples, n_classes) for multiclass classification): The predicted target probabilities corresponding to each sample.
        Raises:
            NotFittedError: If the KNNClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "X"):
            raise NotFittedError("KNNClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        distances: torch.Tensor = self.metric(self.X.unsqueeze(0), X.unsqueeze(1)) # (len(X), len(self.X))
        indicies = distances.topk(self.k, largest=False).indices
        k_labels = self.y[indicies]

        all_classes = torch.unique(self.y)
        probs = torch.zeros((len(X), len(all_classes)))
        for i, labels in enumerate(k_labels):
            classes, counts = torch.unique(labels, return_counts=True)
            prob = torch.tensor([(counts[torch.where(classes == _class)[0]] / len(labels) if _class in classes else 0) for _class in all_classes])
            probs[i] = prob
        if len(all_classes) == 2:  # binary classification
            probs = probs[:, 1]
        return probs
