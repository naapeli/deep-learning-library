import torch

from ....Data.Preprocessing import data_split
from ....Data.Metrics import prob_to_pred
from . import IsotonicRegression
from ..LinearModels import LogisticRegression


class CalibratedClassifier:
    """
    The CalibratedClassifier for probability calibration.

    Args:
        estimator (DLL.Machinelearning classifier): An estimator to calibrate. The estimator must have the predict_proba method defined.
        method (str, optional): The method used for the calibration. Must be one of "logistic" or "isotonic". Defaults to "logistic".
        learning_rate (float, optional): The learning rate for fitting the logistic regression. Must be a positive real number. If method == "isotonic", this parameter is ignored. Defaults to 0.01.
    """
    def __init__(self, estimator, method="logistic", learning_rate=0.01):
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("The estimator must have the predict_proba method defined.")
        if method not in ["isotonic", "logistic"]:
            raise ValueError('The chosen method be one of "logistic" or "isotonic".')
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")

        self.estimator = estimator
        self.method = method
        self.calibrator = LogisticRegression(learning_rate) if method == "logistic" else IsotonicRegression()
    
    def fit(self, X, y, calibration_size=0.2, **kwargs):
        """
        Fits chosen classifier to the input data and the output labels trying to be as accurate as possible about the probabilities of each class. Uses part of the data for the calibration and not for the fitting.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            calibration_size (float, optional): The amount of data used for the model calibration. Defaults to 0.2.
            **kwargs: Key word arguments, which are passed to the fit method of the chosen estimator.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The inpu matrix X and the label matrix y must be PyTorch tensors.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        if not isinstance(calibration_size, int | float) or calibration_size <= 0 or calibration_size >= 1:
            raise ValueError("calibration_size must be in range (0, 1).")
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        X_train, y_train, _, _, X_cal, y_cal = data_split(X, y, train_split=1 - calibration_size, validation_split=0.0)
        self.estimator.fit(X_train, y_train, **kwargs)
        probs = self.estimator.predict_proba(X_cal)
        if self.method == "logistic": probs = probs.unsqueeze(1)
        self.calibrator.fit(self._logit(probs), y_cal, epochs=1000) if self.method == "logistic" else self.calibrator.fit(probs, y_cal)

    def predict_proba(self, X):
        """
        Predicts the calibrated probabilities of the data points.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.

        Returns:
            torch.Tensor of shape (n_samples, n_classes) or (n_samples,): The predicted probabilities.
        """
        probs = self.estimator.predict_proba(X)
        if self.method == "logistic": probs = probs.unsqueeze(1)
        calibrated_probs = self.calibrator.predict_proba(self._logit(probs)) if self.method == "logistic" else self.calibrator.predict(probs)
        return calibrated_probs

    def predict(self, X):
        """
        Predicts the classes of each data point using the calibrated probabilities.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.

        Returns:
            torch.Tensor of shape (n_samples, n_classes) or (n_samples,): The predicted probabilities.
        """
        return prob_to_pred(self.predict_proba(X))

    def _logit(self, X):
        X[X < 1e-6] = 1e-6
        X[X > 1 - 1e-6] = 1 - 1e-6
        return torch.log(X / (1 - X))
