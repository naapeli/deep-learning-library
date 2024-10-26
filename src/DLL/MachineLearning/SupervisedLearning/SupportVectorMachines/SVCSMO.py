import torch
import copy

from ..Kernels import SquaredExponentialCovariance, _Base
from ....Exceptions import NotFittedError


class SVCSMO:
    """
    The support vector machine classifier with the sequential minimal optimization (SMO) quadratic programming solver. The algorithm is based on `this paper <https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf>`_.
    
    
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
        self.tol = 1e-5

        self._from_multi = False
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)

    def fit(self, X, y, epochs=float("inf"), multi_method="ovr", epochs_no_change=5):
        """
        Fits the SVCSMO model to the input data by finding the hyperplane that separates the data with maximum margin.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            epochs (int, optional): The number of rounds we fit the model. Must be a positive integer. Defaults too unlimited number of rounds.
            multi_method (str, optional): The method for multi-class classification. Is ignored for binary classification. Must be one of "ovr" or "ovo". Defaults to "ovr".
            epochs_no_change (int, optional): The number of allowed optimization rounds with no changes. Must be a positive integer. Defaults to 5.

        Returns:
            None
        Raises:
            TypeError: If the input matrix or the label matrix is not a PyTorch tensor, epochs or epochs_no_change is not an integer or multi_method is not a string.
            ValueError: If the input matrix or the label matrix is not the correct shape, epochs or epochs_no_change are not positive or multi_method is not in allowed methods.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the label matrix must be a PyTorch tensor.")
        if not isinstance(multi_method, str):
            raise TypeError("multi_method must be a string.")
        if (not isinstance(epochs, int) and epochs != float("inf")) or not isinstance(epochs_no_change, int):
            raise TypeError("epochs and epochs_no_change must be integers.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        if epochs <= 0 or epochs_no_change <= 0:
            raise ValueError("epochs and epochs_no_change must be positive.")
        if multi_method not in ["ovr", "ovo"]:
            raise ValueError('multi_method must be one of "ovr" or "ovo".')
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        self.n_features = X.shape[1]

        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, multi_method, epochs, epochs_no_change)
        
        self.multiclass = False
        y = torch.where(y == 0, -1, 1)
        self.y = y.to(X.dtype)
        self.X = X
        n = X.shape[0]
        self.K = self._kernel_matrix(X, X)

        self.alpha = torch.zeros(n, dtype=X.dtype)
        self.b = 0

        iter_count = 0
        num_no_change_iter = 0
        while iter_count < epochs:
            num_changed_alphas = 0
            for i in range(n):
                E_i = self._decision_function(self.K[i, :]) - self.y[i]
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # Select j randomly
                    j = self._select_j(i, n)
                    E_j = self._decision_function(self.K[j, :]) - self.y[j]

                    # Compute bounds for α_j
                    if self.y[i] == self.y[j]:
                        L = torch.max(torch.zeros(1), self.alpha[j] + self.alpha[i] - self.C)
                        H = torch.min(torch.tensor(self.C), self.alpha[j] + self.alpha[i])
                    else:
                        L = torch.max(torch.zeros(1), self.alpha[j] - self.alpha[i])
                        H = torch.min(torch.tensor(self.C), self.C + self.alpha[j] - self.alpha[i])
                    if L == H:
                        continue

                    # Compute η (second derivative of the objective function w.r.t. α_j)
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    # Update α_j
                    alpha_j_old = self.alpha[j].clone()
                    self.alpha[j] = self.alpha[j] - (self.y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = torch.clamp(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < self.tol:
                        continue

                    # Update α_i
                    alpha_i_old = self.alpha[i].clone()
                    self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # Update bias term
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                num_no_change_iter += 1
            else:
                num_no_change_iter = 0
            if num_no_change_iter >= epochs_no_change:
                break
            iter_count += 1

        self.is_sv = self.alpha > self.tol

    def _select_j(self, i, n):
        j = i
        while j == i:
            j = torch.randint(0, n, (1,)).item()
        return j

    def _decision_function(self, K):
        return (self.alpha * self.y) @ K + self.b

    def _multi_fit(self, X, y, method, epochs, epochs_no_change):
        self.method = method
        self.n_classes = len(torch.unique(y))
        classes = torch.unique(y)

        if method == "ovr":
            self.classifiers = []
            for label in classes:
                Xs, ys = X, torch.where(y == label, 1, 0)
                classifier = SVCSMO(kernel=self.kernel, C=self.C)
                classifier.fit(Xs, ys, epochs=epochs, epochs_no_change=epochs_no_change)
                self.classifiers.append(classifier)

        elif method == "ovo":
            self.classifiers = []
            for i in range(self.n_classes):
                for j in range(i+1, self.n_classes):
                    class_i, class_j = classes[i], classes[j]
                    indices = (y == class_i) | (y == class_j)
                    X_subset, y_subset = X[indices], y[indices]
                    y_subset = torch.where(y_subset == class_i, 1, 0)
                    classifier = SVCSMO(kernel=self.kernel, C=self.C)
                    classifier.fit(X_subset, y_subset)
                    self.classifiers.append((classifier, class_i, class_j))

    def predict(self, X):
        """
        Applies the fitted SVCSMO model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the SVCSMO model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "multiclass"):
            raise NotFittedError("SVCSMO.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        if self.multiclass:
            return self._multi_predict(X)
        
        score = self._decision_function(self._kernel_matrix(self.X, X))
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
