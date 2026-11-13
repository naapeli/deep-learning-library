import torch

from ..Kernels import RBF, _Base
from ._Primal import _fit_primal
from ....Exceptions import NotFittedError
from ....DeepLearning.Losses import EpsilonInsensitive


class SVR:
    """
    The support vector machine regressor. The model minimises the primal objective :math:`\\frac{1}{2}\\|f\\|^2 + C\\sum_{i=1}^n l(y_i, f(x_i))`, where :math:`f(x) = \\sum_{i=1}^n\\beta_i k(x_i, x) + b` and :math:`l` is the :class:`epsilon-insensitive <DLL.DeepLearning.Losses.EpsilonInsensitive>` or the squared epsilon-insensitive loss. The objective is minimised with Newton's method as in `this paper <https://doi.org/10.1162/neco.2007.19.5.1155>`_. The epsilon-insensitive loss has a zero hessian, so its corners are smoothed by a quadratic function, which is gradually removed during the training. The squared loss is minimised directly and converges in a few iterations.

    Args:
        kernel (:ref:`kernel_section_label`, optional): The non-linearity function for fitting the model. Defaults to RBF.
        C (float or int, optional): A regularization parameter. Defaults to 1. Must be positive real number.
        epsilon (float or int, optional): The width of the tube of no penalty in epsilon-SVR. Must be a positive real number. Defaults to 0.1.
        loss (str, optional): The minimised loss function. Must be one of "epsilon_insensitive" or "squared_epsilon_insensitive". The squared loss penalises large errors more and is faster to optimise. Defaults to "epsilon_insensitive".
    Attributes:
        n_features (int): The number of features. Available after fitting.
        beta (torch.Tensor of shape (n_samples,)): The optimized coefficients of the kernel expansion, which are the differences of the dual coefficients. Available after fitting.
        bias (torch.Tensor): The bias of the model. Available after fitting.
        history (dict[str, torch.Tensor]): The primal objective after every epoch under the key "loss". Available after fitting.
    """
    def __init__(self, kernel=RBF(), C=1, epsilon=0.1, loss="epsilon_insensitive"):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(C, float | int) or C <= 0:
            raise ValueError("C must be must be positive real number.")
        if not isinstance(epsilon, float | int) or epsilon <= 0:
            raise ValueError("epsilon must be must be positive real number.")
        if loss not in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
            raise ValueError('loss must be one of "epsilon_insensitive" or "squared_epsilon_insensitive".')

        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.loss_ = loss
        self.loss = EpsilonInsensitive(epsilon=epsilon, squared=(loss == "squared_epsilon_insensitive"), reduction="sum")

    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)

    def fit(self, X, y, epochs=300):
        """
        Fits the SVR model to the input data by finding the hypertube that contains the data with minimum loss.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            epochs (int, optional): The maximum number of Newton iterations. The training stops earlier once it has converged, which usually takes a few iterations for the squared loss and a few dozen for the epsilon-insensitive loss. Must be a positive integer. Defaults to 300.
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
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        self.X = X
        self.n_features = X.shape[1]
        K = self._kernel_matrix(X, X)
        # The bias is regularised as a part of the kernel expansion, so the targets are centred to keep it small.
        offset = torch.median(y).to(K.dtype)
        self.beta, bias, history = _fit_primal(K, y.to(K.dtype) - offset, self.loss, self.C, epochs)
        self.bias = bias + offset
        self.history = {"loss": history}

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
        if not hasattr(self, "beta"):
            raise NotFittedError("SVR.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        return self.beta @ self._kernel_matrix(self.X, X).to(self.beta.dtype) + self.bias
