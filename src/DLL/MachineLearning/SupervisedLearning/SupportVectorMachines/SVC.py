import torch
import cvxopt
import numpy as np
import copy

from ..Kernels import SquaredExponentialCovariance


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
        self.transform_y = False
        if set(torch.unique(y)) == {0, 1}:
            y[y == 0] = -1
            self.transform_y = True
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
        self.alpha = torch.tensor(np.array(sol["x"]))

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
            self.n_classes = len(torch.unique(y))
            self.classes = torch.unique(y)
            self.classifiers = []
            for i in range(self.n_classes):
                for j in range(i+1, self.n_classes):
                    class_i, class_j = self.classes[i], self.classes[j]
                    indices = (y == class_i) | (y == class_j)
                    X_subset, y_subset = X[indices], y[indices]
                    y_subset = torch.clone(y_subset)
                    y_subset[y_subset == class_j] = -1
                    y_subset[y_subset == class_i] = +1
                    classifier = SVC(kernel=self.kernel, C=self.C)
                    classifier.fit(X_subset, y_subset)
                    self.classifiers.append((classifier, class_i, class_j))
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')

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
            votes = torch.zeros((X.shape[0], self.n_classes))
            for classifier, class_i, class_j in self.classifiers:
                predictions = classifier.predict(X)
                votes[:, class_i] += (predictions == +1)  # If prediction is +1, vote for class_i
                votes[:, class_j] += (predictions == -1)  # If prediction is -1, vote for class_j
            return self.classes[torch.argmax(votes, dim=1)]
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')
