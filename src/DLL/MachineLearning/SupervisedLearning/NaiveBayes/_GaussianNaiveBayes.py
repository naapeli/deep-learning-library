import torch

from ....Exceptions import NotFittedError


class GaussianNaiveBayes:
    """
    The GaussianNaiveBayes classifier model. Applies the Bayes theorem to classify samples with real number features.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        n_classes (int): The number of classes. Available after fitting.
    """
    def fit(self, X, y):
        """
        Fits the GaussianNaiveBayes model to the input data by calculating the prior probabilities and likelihoods.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
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
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        n, self.n_features = X.shape
        self.n_classes = len(vals)
        self.means = torch.zeros((self.n_classes, self.n_features), dtype=X.dtype)
        self.vars = torch.zeros_like(self.means, dtype=X.dtype)
        self.priors = torch.zeros((self.n_classes,), dtype=X.dtype)

        for i in range(self.n_classes):
            X_cls = X[y == i]
            self.means[i] = X_cls.mean(dim=0)
            self.vars[i] = X_cls.var(dim=0)
            self.priors[i] = len(X_cls) / n

    def predict(self, X):
        """
        Applies the fitted GaussianNaiveBayes model to the input data, predicting the labels.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the GaussianNaiveBayes model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "priors"):
            raise NotFittedError("GaussianNaiveBayes.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        posteriors = torch.zeros((self.n_classes, len(X)), dtype=X.dtype)

        for i in range(self.n_classes):
            prior = torch.log(self.priors[i])
            posterior = torch.log(self._pdf(X, self.means[i], self.vars[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        return torch.argmax(posteriors, dim=0)

    def predict_proba(self, X):
        """
        Applies the fitted GaussianNaiveBayes model to the input data, predicting the probabilities.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the GaussianNaiveBayes model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "priors"):
            raise NotFittedError("GaussianNaiveBayes.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        posteriors = torch.zeros((self.n_classes, len(X)), dtype=X.dtype)

        for i in range(self.n_classes):
            prior = torch.log(self.priors[i])
            posterior = torch.log(self._pdf(X, self.means[i], self.vars[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        prob_normalizers = torch.logsumexp(posteriors, dim=0)
        log_probs = posteriors - prob_normalizers
        probs = torch.exp(log_probs).T
        if self.n_classes == 2:  # binary classification
            probs = probs[:, 1]
        return probs

    def _pdf(self, x, mean, var):
        return torch.exp(-(x - mean) ** 2 / (2 * var)) / torch.sqrt(2 * torch.pi * var)
