import torch
import cvxopt
import numpy as np

from ..Kernels import RBF, _Base
from ....Exceptions import NotFittedError


class SVR:
    """
    The support vector machine regressor with a quadratic programming solver.
    
    Args:
        kernel (:ref:`kernel_section_label`, optional): The non-linearity function for fitting the model. Defaults to RBF.
        C (float or int, optional): A regularization parameter. Defaults to 1. Must be positive real number.
        epsilon (float or int, optional): The width of the tube of no penalty in epsilon-SVR. Must be a positive real number.
    Attributes:
        n_features (int): The number of features. Available after fitting.
        alpha (torch.Tensor of shape (n_samples,)): The optimized dual coefficients. Available after fitting.
        alpha_star (torch.Tensor of shape (n_samples,)): The optimized dual coefficients. Available after fitting.
    """
    def __init__(self, kernel=RBF(), C=1, epsilon=0.1):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(C, float | int) or C <= 0:
            raise ValueError("C must be must be positive real number.")
        if not isinstance(epsilon, float | int) or epsilon <= 0:
            raise ValueError("epsilon must be must be positive real number.")

        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y):
        """
        Fits the SVR model to the input data by finding the hypertube that contains the data with minimum loss.

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
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")

        self.y = y.reshape((-1, 1)).to(X.dtype)
        self.X = X
        self.n_features = X.shape[1]
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
        alpha = torch.tensor(np.array(sol["x"]), dtype=torch.float64).squeeze(dim=1)
        self.alpha = alpha[:n]
        self.alpha_star = alpha[n:]

    def predict(self, X):
        """
        Applies the fitted SVR model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the SVR model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "n_features"):
            raise NotFittedError("SVR.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        # Use every datapoint as a support vector, since otherwise the results seem to be bad.
        # This makes no sense to me, but yields the best results.
        bias = (self.y - torch.sum((self.alpha - self.alpha_star) * self._kernel_matrix(self.X, self.X), dim=1)).mean()
        prediction = (self.alpha - self.alpha_star) @ self._kernel_matrix(self.X, X).to(dtype=self.alpha.dtype) + bias
        return prediction
