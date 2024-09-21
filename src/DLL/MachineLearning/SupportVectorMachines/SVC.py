import torch
import cvxopt
import numpy as np
import copy

from ..Kernels import SquaredExponentialCovariance
from ...DeepLearning.Optimisers.ADAM import Adam


"""
The support vector machine classifier with the cvxopt.solver.qp quadratic programming solver. This code is mostly based on https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084#a603 with minor modifications mainly on the multi-class methods.
"""
class SVC:
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1):
        self.kernel = kernel
        self.C = C
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y, multi_method="ovr"):
        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, multi_method)
        self.multiclass = False
        if set(torch.unique(y)) == {0, 1}: y[y == 0] = -1
        self.y = torch.where(y <= 0, -1, 1).reshape((-1, 1)).to(X.dtype)
        self.transform_y = not torch.all(self.y == y)
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
        self.alpha = torch.tensor(np.array(sol["x"]), )

        self.is_sv = ((self.alpha - 1e-3 > 0) & (self.alpha <= self.C)).squeeze()
        self.margin_sv = torch.argmax(((0 < self.alpha - 1e-3) & (self.alpha < self.C - 1e-3)).to(torch.int32))

    def _multi_fit(self, X, y, method):
        assert hasattr(self, "multiclass"), "Use SVC.fit(X, y) rather than SVC._multi_fit(X, y) even if your data is multidimensional."
        self.method = method
        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classes = torch.unique(y)
            self.classifiers = []
            for label in self.classes:
                Xs, ys = X, copy.deepcopy(y)
                ys[ys != label], ys[ys == label] = -1, +1
                classifier = SVC(kernel=self.kernel, C=self.C)
                classifier.fit(Xs, ys)
                self.classifiers.append(classifier)

        elif method == "ovo":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def predict(self, X, _from_multi=False):
        assert hasattr(self, "multiclass"), "SVC.fit(X, y) must be called before predicting."
        if self.multiclass:
            self.flag = None
            return self._multi_predict(X)
        x_s, y_s = self.X[self.margin_sv, torch.newaxis], self.y[self.margin_sv]
        alpha_sv, y_sv, X_sv = self.alpha[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
        bias = y_s - torch.sum(alpha_sv * y_sv * self._kernel_matrix(X_sv, x_s), dim=0)
        score = torch.sum(alpha_sv * y_sv * self._kernel_matrix(X_sv, X), dim=0) + bias
        if _from_multi:
            return score
        return ((torch.sign(score) + 1) / 2).to(torch.int32) if self.transform_y else torch.sign(score).to(torch.int32)

    def _multi_predict(self, X):
        assert hasattr(self, "flag"), "Use SVC.predict(X, y) rather than SVC._multi_predict(X, y) even if your data is multidimensional."
        if self.method == "ovr":
            predictions = torch.zeros((X.shape[0], self.n_classes))
            for i, classifier in enumerate(self.classifiers):
                predictions[:, i] = classifier.predict(X, _from_multi=True)
            return self.classes[torch.argmax(predictions, dim=1)]

        elif self.method == "ovo":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


"""
The support vector machine classifier with the sequential minimal optimization (SMO) quadratic programming solver.
"""
class SVCSMO:
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1):
        self.kernel = kernel
        self.C = torch.tensor(C)
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y, epochs=float("inf"), multi_method="ovr", tol=1e-5, epochs_no_change=5):
        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, multi_method, epochs, tol, epochs_no_change)
        self.multiclass = False
        if set(torch.unique(y)) == {0, 1}: y[y == 0] = -1
        self.y = torch.where(y <= 0, -1, 1).reshape((-1, 1)).to(X.dtype)
        self.transform_y = not torch.all(self.y == y)
        self.X = X
        n = X.shape[0]
        self.K = self._kernel_matrix(X, X)

        self.alpha = torch.zeros(n, 1, dtype=X.dtype)
        self.b = 0

        iter_count = 0
        num_no_change_iter = 0
        while iter_count < epochs:
            num_changed_alphas = 0
            for i in range(n):
                E_i = self._decision_function(self.K[i, :].unsqueeze(-1)) - self.y[i]
                if (self.y[i] * E_i < -tol and self.alpha[i] < self.C) or (self.y[i] * E_i > tol and self.alpha[i] > 0):
                    # Select j randomly
                    j = self._select_j(i, n)
                    E_j = self._decision_function(self.K[j, :].unsqueeze(-1)) - self.y[j]

                    # Compute bounds for α_j
                    if self.y[i] == self.y[j]:
                        L = max(torch.zeros(1), self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = max(torch.zeros(1), self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
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

                    if abs(self.alpha[j] - alpha_j_old) < tol:
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

        self.is_sv = (self.alpha > tol).squeeze()

    def _select_j(self, i, n):
        j = i
        while j == i:
            j = torch.randint(0, n, (1,)).item()
        return j

    def _decision_function(self, K):
        return (self.alpha * self.y).T @ K + self.b

    def _multi_fit(self, X, y, method, epochs, tol, epochs_no_change):
        assert hasattr(self, "multiclass"), "Use SVC.fit(X, y) rather than SVC._multi_fit(X, y) even if your data is multidimensional."
        self.method = method
        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classes = torch.unique(y)
            self.classifiers = []
            for label in self.classes:
                Xs, ys = X, copy.deepcopy(y)
                ys[ys != label], ys[ys == label] = -1, +1
                classifier = SVCSMO(kernel=self.kernel, C=self.C.item())
                classifier.fit(Xs, ys, epochs=epochs, tol=tol, epochs_no_change=epochs_no_change)
                self.classifiers.append(classifier)

        elif method == "ovo":
            raise NotImplementedError('The "ovo" method is not yet implemented.')
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')

    def predict(self, X, _from_multi=False):
        assert hasattr(self, "multiclass"), "SVC.fit(X, y) must be called before predicting."
        if self.multiclass:
            self.flag = None
            return self._multi_predict(X)
        score = self._decision_function(self._kernel_matrix(self.X, X)).squeeze()
        if _from_multi:
            return score
        return ((torch.sign(score) + 1) / 2).to(torch.int32) if self.transform_y else torch.sign(score).to(torch.int32)

    def _multi_predict(self, X):
        assert hasattr(self, "flag"), "Use SVC.predict(X, y) rather than SVC._multi_predict(X, y) even if your data is multidimensional."
        if self.method == "ovr":
            predictions = torch.zeros((X.shape[0], self.n_classes))
            for i, classifier in enumerate(self.classifiers):
                predictions[:, i] = classifier.predict(X, _from_multi=True)
            return self.classes[torch.argmax(predictions, dim=1)]

        elif self.method == "ovo":
            raise NotImplementedError('The "ovo" method is not yet implemented.')
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')
