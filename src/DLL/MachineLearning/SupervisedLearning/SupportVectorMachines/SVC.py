import torch
import cvxopt
import numpy as np

from ..Kernels import SquaredExponentialCovariance, _Base
from ....Exceptions import NotFittedError


class SVC:
    """
    The support vector machine classifier. "cvxopt"-optimization method's implementation is largly based on `this article <https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084#a603>`_. The "smo"-optimization algorithm is based on `this paper <https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf>`_.

    Args:
        kernel (:ref:`kernel_section_label`, optional): The non-linearity function for fitting the model. Defaults to SquaredExponentialCovariance.
        C (float or int, optional): A regularization parameter. Defaults to 1. Must be positive real number.
        opt_method (str, optional): The method that will be used in the optimization step. Must be in ["cvxopt", "smo", "coord_ascent"]. Defaults to "coord_ascent". For optimal results one should experiment with every optimization method, but as a rule of thumb, "coord_ascent" is the most efficient.
    Attributes:
        n_features (int): The number of features. Available after fitting.
        alpha (torch.Tensor of shape (n_samples,)): The optimized dual coefficients. Available after fitting.
    """
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1, opt_method="coord_ascent"):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(C, float | int) or C <= 0:
            raise ValueError("C must be must be positive real number.")
        if opt_method not in ["cvxopt", "smo", "coord_ascent"]:
            raise ValueError('opt_method must be in ["cvxopt", "smo", "coord_ascent"].')

        self.kernel = kernel
        self.C = C
        self.opt_method = opt_method

        self._from_multi = False
        self.tol = 1e-5

    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y, epochs=10, multi_method="ovr"):
        """
        Fits the SVC model to the input data by finding the hyperplane that separates the data with maximum margin. It uses the optimization method defined in the constructor of the class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            epochs (int, optional): The number of iterations the training loop runs. Defaults to 100. Must be a positive integer. Is ignored for opt_method="cvxopt".
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
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if multi_method not in ["ovr", "ovo"]:
            raise ValueError('multi_method must be one of "ovr" or "ovo".')
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        self.n_features = X.shape[1]

        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, epochs, multi_method)
        
        self.multiclass = False
        y = torch.where(y == 0, -1, 1)
        self.y = y
        self.X = X
        n = X.shape[0]
        K = self._kernel_matrix(X, X)
        
        if self.opt_method == "cvxopt":
            y_vec = y.reshape((-1, 1)).to(X.dtype)

            P = cvxopt.matrix((y_vec @ y_vec.T * K).numpy())
            q = cvxopt.matrix(-np.ones((n, 1)))
            A = cvxopt.matrix((y_vec.T).numpy())
            b = cvxopt.matrix(np.zeros(1))
            G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
            h = cvxopt.matrix(np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * self.C)))
            
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            self.alpha = torch.tensor(np.array(sol["x"])).squeeze()
        
        elif self.opt_method == "coord_ascent":
            alpha = torch.zeros(n, dtype=X.dtype)

            for _ in range(epochs):
                # process all sample examples sequentially
                for i in range(n):
                    new_val = (1 - y[i] * ((alpha * y * K[i]).sum() - alpha[i] * y[i] * K[i, i])) / K[i, i]
                    new_val = min(self.C, max(new_val, 0))
                    alpha[i] = new_val
            self.alpha = alpha

        elif self.opt_method == "smo":
            self.alpha = torch.zeros(n, dtype=X.dtype)
            self.b = 0
            epochs_no_change = 2

            iter_count = 0
            num_no_change_iter = 0
            while iter_count < epochs:
                num_changed_alphas = 0
                for i in range(n):
                    E_i = (self.alpha * self.y) @ K[i, :] + self.b - self.y[i]
                    if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                        # Select j randomly
                        j = self._select_j(i, n)
                        E_j = (self.alpha * self.y) @ K[j, :] + self.b - self.y[j]

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
                        eta = 2 * K[i, j] - K[i, i] - K[j, j]
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
                        b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                        b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

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

        self.is_sv = (self.alpha - self.tol > 0) & (self.alpha <= self.C)
        self.margin_sv = torch.argmax(((0 < self.alpha - self.tol) & (self.alpha < self.C - self.tol)).to(torch.int32))

        if self.opt_method != "smo":
            x_s, y_s = self.X[self.margin_sv, torch.newaxis], self.y[self.margin_sv]
            alpha_sv, y_sv, X_sv = self.alpha[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
            self.b = y_s - torch.sum((alpha_sv * y_sv).unsqueeze(-1) * self._kernel_matrix(X_sv, x_s), dim=0)
    
    def _select_j(self, i, n):
        j = i
        while j == i:
            j = torch.randint(0, n, (1,)).item()
        return j

    def _multi_fit(self, X, y, epochs, method):
        self.method = method
        classes = torch.unique(y)

        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classifiers = []
            for label in classes:
                Xs, ys = X, torch.where(y == label, 1, 0)
                classifier = SVC(kernel=self.kernel, C=self.C, opt_method=self.opt_method)
                classifier.fit(Xs, ys, epochs)
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
                    classifier = SVC(kernel=self.kernel, C=self.C, opt_method=self.opt_method)
                    classifier.fit(X_subset, y_subset, epochs)
                    self.classifiers.append((classifier, class_i, class_j))
        
    def _decision_function(self, K):
        return (self.alpha * self.y) @ K + self.b

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
