import torch
import cvxopt
import numpy as np

from ..Kernels import SquaredExponentialCovariance, _Base
from ....Exceptions import NotFittedError


class SVC:
    """
    The support vector machine classifier with a quadratic programming solver. This implementation is largly based on `this article <https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084#a603>`_ with modifications mainly on the multi-class methods.
    
    Args:
        kernel (:ref:`kernel_section_label`, optional): The non-linearity function for fitting the model. Defaults to SquaredExponentialCovariance.
        C (float or int, optional): A regularization parameter. Defaults to 1. Must be positive real number.
    Attributes:
        n_features (int): The number of features. Available after fitting.
        alpha (torch.Tensor of shape (n_samples,)): The optimized dual coefficients. Available after fitting.
    """
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(C, float | int) or C <= 0:
            raise ValueError("C must be must be positive real number.")

        self.kernel = kernel
        self.C = C

        self._from_multi = False
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y, multi_method="ovr"):
        """
        Fits the SVC model to the input data by finding the hyperplane that separates the data with maximum margin.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            multi_method (str, optional): The method for multi-class classification. Is ignored for binary classification. Must be one of "ovr" or "ovo". Defaults to "ovr".
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the label matrix is not a PyTorch tensor or multi_method is not a string.
            ValueError: If the input matrix or the label matrix is not the correct shape or multi_method is not in allowed methods.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the label matrix must be a PyTorch tensor.")
        if not isinstance(multi_method, str):
            raise TypeError("multi_method must be a string.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        if multi_method not in ["ovr", "ovo"]:
            raise ValueError('multi_method must be one of "ovr" or "ovo".')
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        self.n_features = X.shape[1]

        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, multi_method)
        
        self.multiclass = False
        y = torch.where(y == 0, -1, 1)
        self.y = y.reshape((-1, 1)).to(X.dtype)
        self.X = X
        n = X.shape[0]
        K = self._kernel_matrix(X, X)

        P = cvxopt.matrix((self.y @ self.y.T * K).numpy())
        q = cvxopt.matrix(-np.ones((n, 1)))
        A = cvxopt.matrix((self.y.T).numpy())
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = cvxopt.matrix(np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * self.C)))
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = torch.tensor(np.array(sol["x"])).squeeze()

        self.is_sv = (self.alpha - 1e-3 > 0) & (self.alpha <= self.C)
        self.margin_sv = torch.argmax(((0 < self.alpha - 1e-3) & (self.alpha < self.C - 1e-3)).to(torch.int32))

    def _multi_fit(self, X, y, method):
        self.method = method
        classes = torch.unique(y)

        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classifiers = []
            for label in classes:
                Xs, ys = X, torch.where(y == label, 1, 0)
                classifier = SVC(kernel=self.kernel, C=self.C)
                classifier.fit(Xs, ys)
                self.classifiers.append(classifier)

        elif method == "ovo":
            self.n_classes = len(torch.unique(y))
            self.classifiers = []
            for i in range(self.n_classes):
                for j in range(i+1, self.n_classes):
                    class_i, class_j = classes[i], classes[j]
                    indices = (y == class_i) | (y == class_j)
                    X_subset, y_subset = X[indices], y[indices]
                    y_subset = torch.where(y_subset == class_i, 1, 0)
                    classifier = SVC(kernel=self.kernel, C=self.C)
                    classifier.fit(X_subset, y_subset)
                    self.classifiers.append((classifier, class_i, class_j))

    def predict(self, X):
        """
        Applies the fitted SVC model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the SVC model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "multiclass"):
            raise NotFittedError("SVC.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        if self.multiclass:
            return self._multi_predict(X)
        
        x_s, y_s = self.X[self.margin_sv, torch.newaxis], self.y.squeeze(dim=1)[self.margin_sv]
        alpha_sv, y_sv, X_sv = self.alpha[self.is_sv], self.y.squeeze(dim=1)[self.is_sv], self.X[self.is_sv]
        bias = y_s - torch.sum((alpha_sv * y_sv).unsqueeze(-1) * self._kernel_matrix(X_sv, x_s), dim=0)  # .unsqueeze(-1) for appropriate broadcasting
        score = torch.sum((alpha_sv * y_sv).unsqueeze(-1) * self._kernel_matrix(X_sv, X), dim=0) + bias  # .unsqueeze(-1) for appropriate broadcasting
        if self._from_multi:
            return score
        return ((torch.sign(score) + 1) / 2).to(torch.int32)

    def _multi_predict(self, X):
        if self.method == "ovr":
            predictions = torch.zeros((X.shape[0], self.n_classes))
            for i, classifier in enumerate(self.classifiers):
                classifier._from_multi = True
                predictions[:, i] = classifier.predict(X)
            return torch.argmax(predictions, dim=1)

        elif self.method == "ovo":
            votes = torch.zeros((X.shape[0], self.n_classes))
            for classifier, class_i, class_j in self.classifiers:
                predictions = classifier.predict(X)
                votes[:, class_i] += (predictions == 1)  # If prediction is 1, vote for class_i
                votes[:, class_j] += (predictions == 0)  # If prediction is 0, vote for class_j
            return torch.argmax(votes, dim=1)
