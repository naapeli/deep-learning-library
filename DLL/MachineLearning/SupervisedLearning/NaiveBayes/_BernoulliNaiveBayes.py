import torch

from ....Exceptions import NotFittedError


class BernoulliNaiveBayes:
    """
    The BernoulliNaiveBayes classifier model. Applies the Bayes theorem to classify samples with binary features.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        n_classes (int): The number of classes. Available after fitting.
    """
    def fit(self, X, y, alpha=1):
        """
        Fits the BernoulliNaiveBayes model to the input data by calculating the prior probabilities and likelihoods.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature. Must contain only binary values.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            alpha (float | int): Laplacian smoothing parameter. Must be non-negative. For no smoothing, alpha is set to zero. Defaults to 1.
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
        if set(torch.unique(X).numpy()) != {0, 1}:
            raise ValueError("The features must be labeled 0 and 1. If your features are not binary, consider other NaiveBayes models.")
        if not isinstance(alpha, int | float) or alpha < 0:
            raise ValueError("alpha must be a non-negative real number.")

        self.n_features = X.shape[1]
        self.n_classes = len(vals)
        self.priors = torch.zeros((self.n_classes,), dtype=torch.float32)
        self.likelihoods = torch.zeros((self.n_classes, X.shape[1]), dtype=torch.float32)

        for i in range(self.n_classes):
            X_cls = X[y == i]
            self.priors[i] = len(X_cls) / len(y)
            self.likelihoods[i] = (X_cls.sum(dim=0) + alpha) / (len(X_cls) + alpha * X.shape[1]) # laplace smoothing

    def predict(self, X):
        """
        Applies the fitted BernoulliNaiveBayes model to the input data, predicting the labels.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the BernoulliNaiveBayes model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "priors"):
            raise NotFittedError("BernoulliNaiveBayes.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        if set(torch.unique(X).numpy()) != {0, 1}:
            raise ValueError("The features must be labeled 0 and 1. If your features are not binary, consider other NaiveBayes models.")
        
        posteriors = torch.zeros((self.n_classes, len(X)), dtype=torch.float32)

        for i in range(self.n_classes):
            prior = torch.log(self.priors[i])
            posterior = (X * torch.log(self.likelihoods[i]) + (1 - X) * torch.log(1 - self.likelihoods[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        return torch.argmax(posteriors, dim=0)

    def predict_proba(self, X):
        """
        Applies the fitted BernoulliNaiveBayes model to the input data, predicting the labels.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the BernoulliNaiveBayes model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "priors"):
            raise NotFittedError("BernoulliNaiveBayes.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        if set(torch.unique(X).numpy()) != {0, 1}:
            raise ValueError("The features must be labeled 0 and 1. If your features are not binary, consider other NaiveBayes models.")

        posteriors = torch.zeros((self.n_classes, len(X)), dtype=torch.float32)

        for i in range(self.n_classes):
            prior = torch.log(self.priors[i])
            posterior = (X * torch.log(self.likelihoods[i]) + (1 - X) * torch.log(1 - self.likelihoods[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        prob_normalizers = torch.logsumexp(posteriors, dim=0)
        log_probs = posteriors - prob_normalizers
        probs = torch.exp(log_probs).T
        if self.n_classes == 2:  # binary classification
            probs = probs[:, 1]
        return probs
