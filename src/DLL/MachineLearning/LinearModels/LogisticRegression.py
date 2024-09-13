import torch

from ...DeepLearning.Losses.BCE import bce
from ...DeepLearning.Layers.Activations.Sigmoid import Sigmoid


class LogisticRegression:
    def __init__(self, iterations=100, learning_rate=0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss = bce()
        self.activation = Sigmoid()

    def fit(self, X, Y):
        batch_size, features = X.shape
        self.weights = torch.zeros(features)
        self.bias = 0
        for i in range(self.iterations):
            y = X @ self.weights + self.bias
            predictions = self.activation.forward(y)
            bce_derivative = self.loss.gradient(predictions, Y)
            dCdy = self.activation.backward(bce_derivative, derivative=True)
            dCdweights = X.T @ dCdy
            dCdbias = torch.mean(dCdy)
            self.weights -= self.learning_rate * dCdweights
            self.bias -= self.learning_rate * dCdbias

    def predict(self, X):
        assert hasattr(self, "weights"), "LogisticRegression.fit() must be called before attempting to predict"
        return X @ self.weights + self.bias
