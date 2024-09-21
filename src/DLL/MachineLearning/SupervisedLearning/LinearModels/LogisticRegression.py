import torch
from math import floor

from ....DeepLearning.Losses.BCE import bce
from ....DeepLearning.Losses.CCE import cce
from ....DeepLearning.Layers.Activations.Sigmoid import Sigmoid
from ....DeepLearning.Layers.Activations.SoftMax import SoftMax
from ....Data.Metrics import accuracy
from ....Data.DataReader import DataReader
from ....DeepLearning.Optimisers.ADAM import Adam


class LogisticRegression:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def fit(self, X, Y, val_data=None, epochs=100, optimiser=None, callback_frequency=1, metrics=["loss"], batch_size=None, shuffle_every_epoch=True, shuffle_data=True):
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
                    values = self._calculate_metrics(data=(X, Y), val_data=val_data)
                    for metric, value in values.items():
                        history[metric][int(epoch / callback_frequency)] = value
                    print(f"Epoch: {epoch + 1} - Metrics: {self._round_dictionary(values)}")
        return history
    
    def _round_dictionary(self, values):
        return {key: "{:0.4f}".format(value) for key, value in values.items()}
    
    def _calculate_metrics(self, data=None, val_data=None):
        values = {}
        if data:
            X, Y = data
            predictions = self.predict(X)
        if val_data:
            x_val, y_val = val_data
            val_predictions = self.predict(x_val)
        for metric in self.metrics:
            if val_data and metric[:3] == "val":
                if metric == "val_loss":
                    metric_value = self.loss.loss(val_predictions, y_val).item()
                elif metric == "val_accuracy":
                    metric_value = accuracy(val_predictions, y_val)
            elif metric == "loss":
                metric_value = self.loss.loss(predictions, Y).item()
            elif metric == "accuracy":
                metric_value = accuracy(predictions, Y)
            else:
                print(f"Metric {metric} not implemented")

            values[metric] = metric_value
        return values

    def predict(self, X):
        assert hasattr(self, "weights"), "LogisticRegression.fit() must be called before attempting to predict"
        return self.activation.forward(X @ self.weights + self.bias)
