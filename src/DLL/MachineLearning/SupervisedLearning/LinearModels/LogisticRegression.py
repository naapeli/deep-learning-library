import torch
from math import floor

from ....DeepLearning.Losses.BCE import bce
from ....DeepLearning.Losses.CCE import cce
from ....DeepLearning.Layers.Activations.Sigmoid import Sigmoid
from ....DeepLearning.Layers.Activations.SoftMax import SoftMax
from ....Data.Metrics import calculate_metrics, _round_dictionary
from ....Data.DataReader import DataReader
from ....DeepLearning.Optimisers.ADAM import Adam


class LogisticRegression:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def fit(self, X, Y, val_data=None, epochs=100, optimiser=None, callback_frequency=1, metrics=["loss"], batch_size=None, shuffle_every_epoch=True, shuffle_data=True, verbose=False):
        batch_size, features = X.shape
        if len(Y.shape) == 1:
            # normal logistic regression
            self.loss = bce()
            self.activation = Sigmoid()
            weight_shape = (features,)
        elif len(Y.shape) == 2:
            # multiple logistic regression
            self.loss = cce()
            self.activation = SoftMax()
            weight_shape = (features, Y.shape[1])

        self.metrics = metrics
        history = {metric: torch.zeros(floor(epochs / callback_frequency)) for metric in metrics}
        batch_size = len(X) if batch_size is None else batch_size
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)

        self.weights = torch.randn(weight_shape)
        self.bias = torch.zeros((1,)) if len(weight_shape) == 1 else torch.zeros(Y.shape[1])
        optimiser = Adam(self.learning_rate) if optimiser is None else optimiser
        optimiser.initialise_parameters([self.weights, self.bias])

        for epoch in range(epochs):
            for x, y in data_reader.get_data():
                y_linear = x @ self.weights + self.bias
                predictions = self.activation.forward(y_linear)
                bce_derivative = self.loss.gradient(predictions, y)
                dCdy = self.activation.backward(bce_derivative)
                dCdweights = x.T @ dCdy
                dCdbias = torch.mean(dCdy, dim=0, keepdim=(len(Y.shape) == 1))
                self.weights.grad = dCdweights
                self.bias.grad = dCdbias
                optimiser.update_parameters()
            if epoch % callback_frequency == 0:
                values = calculate_metrics(data=(self.predict(X), Y), metrics=self.metrics, loss=self.loss.loss)
                if val_data is not None:
                    val_values = calculate_metrics(data=(self.predict(val_data[0]), val_data[1]), metrics=self.metrics, loss=self.loss.loss)
                    values |= val_values
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        return history

    def predict(self, X):
        assert hasattr(self, "weights"), "LogisticRegression.fit() must be called before attempting to predict"
        return self.activation.forward(X @ self.weights + self.bias)
