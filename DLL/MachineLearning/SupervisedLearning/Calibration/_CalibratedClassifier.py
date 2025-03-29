import torch

from ....Data.Preprocessing import data_split
from ....Data.Metrics import prob_to_pred
from . import IsotonicRegression
from ..LinearModels import LogisticRegression


class CalibratedClassifier:
    def __init__(self, estimator, method="logistic", learning_rate=0.01):
        self.estimator = estimator
        self.method = method
        self.calibrator = LogisticRegression(learning_rate) if method == "logistic" else IsotonicRegression()
    
    def fit(self, X, y, calibration_size=0.2, **kwargs):
        X_train, y_train, _, _, X_cal, y_cal = data_split(X, y, train_split=1 - calibration_size, validation_split=0.0)
        self.estimator.fit(X_train, y_train, **kwargs)
        probs = self.estimator.predict_proba(X_cal)
        if self.method == "logistic": probs = probs.unsqueeze(1)
        self.calibrator.fit(self._logit(probs), y_cal, epochs=1000) if self.method == "logistic" else self.calibrator.fit(probs, y_cal)
        # if self.method == "logistic":
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.scatter(self._logit(probs), y_cal)
        #     plt.scatter(self._logit(probs), self.calibrator.predict_proba(self._logit(probs)))
        #     plt.show()

    def predict_proba(self, X):
        probs = self.estimator.predict_proba(X)
        if self.method == "logistic": probs = probs.unsqueeze(1)
        calibrated_probs = self.calibrator.predict_proba(self._logit(probs)) if self.method == "logistic" else self.calibrator.predict(probs)
        return calibrated_probs

    def predict(self, X):
        return prob_to_pred(self.predict_proba(X))

    def _logit(self, X):
        X[X < 1e-6] = 1e-6
        X[X > 1 - 1e-6] = 1 - 1e-6
        return torch.log(X / (1 - X))
