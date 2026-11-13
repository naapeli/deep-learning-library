import torch

from ..Kernels import RBF, _Base
from ._Primal import _fit_primal
from ....Exceptions import NotFittedError
from ....DeepLearning.Losses import Hinge


class SVC:
    """
    The support vector machine classifier. The model minimises the primal objective :math:`\frac{1}{2}\|f\|^2 + C\sum_{i=1}^n l(y_i, f(x_i))`, where :math:`f(x) = \sum_{i=1}^n\beta_i k(x_i, x) + b` and :math:`l` is the :class:`hinge <DLL.DeepLearning.Losses.Hinge>` or the squared hinge loss. The objective is minimised with Newton's method as in `this paper <https://doi.org/10.1162/neco.2007.19.5.1155>`_. The hinge loss has a zero hessian, so its corner is smoothed by a quadratic function, which is gradually removed during the training. The squared hinge loss is minimised directly and converges in a few iterations.

    Args:
        kernel (:ref:`kernel_section_label`, optional): The non-linearity function for fitting the model. Defaults to RBF.
        C (float or int, optional): A regularization parameter. Defaults to 1. Must be positive real number.
        loss (str, optional): The minimised loss function. Must be one of "hinge" or "squared_hinge". The hinge loss gives the classical support vector machine, while the squared hinge loss penalises large violations more and is faster to optimise. Defaults to "hinge".
    Attributes:
        n_features (int): The number of features. Available after fitting.
        alpha (torch.Tensor of shape (n_samples,)): The optimized dual coefficients, i.e. :math:`\beta_iy_i`. Available after fitting a binary classifier.
        history (dict[str, torch.Tensor]): The primal objective after every epoch under the key "loss". Available after fitting a binary classifier.
    """
    def __init__(self, kernel=RBF(), C=1, loss="hinge"):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(C, float | int) or C <= 0:
            raise ValueError("C must be must be positive real number.")
        if loss not in ["hinge", "squared_hinge"]:
            raise ValueError('loss must be one of "hinge" or "squared_hinge".')

        self.kernel = kernel
        self.C = C
        self.loss_ = loss
        self.loss = Hinge(squared=(loss == "squared_hinge"), reduction="sum")

        self._from_multi = False

    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype)
    
    def fit(self, X, y, epochs=300, multi_method="ovr"):
        """
        Fits the SVC model to the input data by finding the hyperplane that separates the data with maximum margin.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            epochs (int, optional): The maximum number of Newton iterations. The training stops earlier once it has converged, which usually takes a few iterations for the squared hinge loss and a few dozen for the hinge loss. Must be a positive integer. Defaults to 300.
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
        self.X = X
        self.y = torch.where(y == 0, -1, 1)
        K = self._kernel_matrix(X, X)
        # the hinge loss expects the labels 0 and 1
        beta, self.b, history = _fit_primal(K, y.to(K.dtype), self.loss, self.C, epochs)
        self.alpha = beta * self.y
        self.history = {"loss": history}

    def _multi_fit(self, X, y, epochs, method):
        self.method = method
        classes = torch.unique(y)

        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classifiers = []
            for label in classes:
                Xs, ys = X, torch.where(y == label, 1, 0)
                classifier = SVC(kernel=self.kernel, C=self.C, loss=self.loss_)
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
                    classifier = SVC(kernel=self.kernel, C=self.C, loss=self.loss_)
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
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data.

        Returns:
            torch.Tensor of shape (n_samples, n_classes) or (n_samples,): The predicted probabilities.
        """
        if not hasattr(self, "multiclass"):
            raise NotFittedError("SVC.fit() must be called before predicting.")

        if not self.multiclass:
            scores = self._decision_function(self._kernel_matrix(self.X, X))
            return 1 / (1 + torch.exp(-scores))

        votes = torch.zeros((X.shape[0], self.n_classes))
        if self.method == "ovr":
            for i, classifier in enumerate(self.classifiers):
                scores = classifier._decision_function(classifier._kernel_matrix(classifier.X, X))
                proba = 1 / (1 + torch.exp(-scores))
                votes[:, i] = proba
        elif self.method == "ovo":
            pairwise_counts = torch.zeros((self.n_classes,), dtype=torch.float32)
            for classifier, class_i, class_j in self.classifiers:
                scores = classifier._decision_function(classifier._kernel_matrix(classifier.X, X))
                proba = 1 / (1 + torch.exp(-scores))
                votes[:, class_i] += proba
                votes[:, class_j] += 1 - proba
                pairwise_counts[class_i] += 1
                pairwise_counts[class_j] += 1
            votes /= pairwise_counts

        return votes / votes.sum(dim=1, keepdim=True)
