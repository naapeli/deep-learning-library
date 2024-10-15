import torch
import cvxopt
import numpy as np

from ..Kernels import SquaredExponentialCovariance


"""
The support vector machine regressor with the cvxopt.solver.qp quadratic programming solver.
"""
class SVR:
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.tol = 1e-5
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y):
        self.y = y.reshape((-1, 1)).to(X.dtype)
        self.X = X
        n = X.shape[0]
        K = self._kernel_matrix(X, X).numpy()

        P = cvxopt.matrix(np.block([[K, -K], [-K, K]]).tolist())  # [[K, -K], [-K, K]]
        q = cvxopt.matrix(np.hstack([self.epsilon - y, self.epsilon + y]).tolist())
        A = cvxopt.matrix(np.vstack([np.ones((n, 1)), -np.ones((n, 1))]).tolist())
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.hstack([-np.eye(2 * n), np.eye(2 * n)]).tolist())
        h = cvxopt.matrix(np.hstack([np.zeros((2 * n,)), np.ones((2 * n,)) * self.C]).tolist())

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = torch.tensor(np.array(sol["x"]), dtype=torch.float64)
        self.alpha = alpha[:n]
        self.alpha_star = alpha[n:]

    def predict(self, X):
        assert hasattr(self, "alpha"), "SVR.fit(X, y) must be called before predicting."
        # Use every datapoint as a support vector, since otherwise the results seem to be bad.
        # This makes no sense to me, but yields the best results.
        bias = (self.y - torch.sum((self.alpha - self.alpha_star) * self._kernel_matrix(self.X, self.X), dim=1)).mean()
        prediction = (self.alpha - self.alpha_star).T @ self._kernel_matrix(self.X, X).to(dtype=self.alpha.dtype) + bias
        return prediction.squeeze()
