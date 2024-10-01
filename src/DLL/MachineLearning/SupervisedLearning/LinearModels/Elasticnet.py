import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import scienceplots
from math import floor

from ....DeepLearning.Losses.MSE import mse
from ....Data.Metrics import calculate_metrics, _round_dictionary
from ....Data.DataReader import DataReader
from ....DeepLearning.Optimisers.ADAM import Adam


class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, loss=mse()):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.loss = loss

    def fit(self, X, Y, val_data=None, epochs=100, optimiser=None, callback_frequency=1, metrics=["loss"], batch_size=None, shuffle_every_epoch=True, shuffle_data=True, verbose=False):
        if len(X.shape) == 1: X = X.unsqueeze(1)
        self.X = X
        self.Y = Y
        features = X.shape[1]
        self.metrics = metrics
        history = {metric: torch.zeros(floor(epochs / callback_frequency)) for metric in metrics}
        batch_size = len(X) if batch_size is None else batch_size
        data_reader = DataReader(X, Y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)

        self.weights = torch.randn((features,))
        self.bias = torch.zeros((1,))
        optimiser = Adam(self.learning_rate) if optimiser is None else optimiser
        optimiser.initialise_parameters([self.weights, self.bias])

        for epoch in range(epochs):
            for x, y in data_reader.get_data():
                predictions = self.predict(x)
                dCdy = self.loss.gradient(predictions, y)
                dCdweights = (x.T @ dCdy) + self.alpha * self.l1_ratio * torch.sign(self.weights) + self.alpha * (1 - self.l1_ratio) * self.weights
                dCdbias = dCdy.mean(dim=0, keepdim=True)
                self.weights.grad = dCdweights
                self.bias.grad = dCdbias
                optimiser.update_parameters()
            if epoch % callback_frequency == 0:
                values = calculate_metrics(data=(self.predict(X), Y), metrics=self.metrics, loss=self.loss.loss)
                if val_data is not None:
                    val_values = calculate_metrics(data=(self.predict(val_data[0]), val_data[1]), metrics=self.metrics, loss=self.loss.loss, validation=True)
                    values |= val_values
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        self.residuals = Y - self.predict(X)
        return history

    def predict(self, X):
        assert hasattr(self, "weights"), "ElasticNetRegression.fit(x, y) must be called before predicting"
        if len(X.shape) == 1: X = X.unsqueeze(1)
        return X @ self.weights + self.bias

    def plot(self, axis_labels=("x", "y", "z"), title="ElasticNet regression", model_label="Model", scatter_label="Datapoints", model_color=None, scatter_color=None, model_opacity=None):
        assert hasattr(self, "weights"), "ElasticNetRegression.fit(x, y) must be called before plotting"
        match self.X.shape[1]:
            case 1:
                x = self.X[:, 0]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(x, self.Y, ".", color=scatter_color, label=scatter_label)
                ax.plot(x, self.predict(self.X), color=model_color, alpha=model_opacity, label=model_label)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
            case 2:
                x = self.X[:, 0]
                y = self.X[:, 1]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, self.Y, label=scatter_label, color=scatter_color)
                x = torch.linspace(torch.min(x), torch.max(x), 2)
                y = torch.linspace(torch.min(y), torch.max(y), 2)
                XX, YY = torch.meshgrid(x, y, indexing="xy")
                X = XX.flatten()
                Y = YY.flatten()
                X_input = torch.stack((X, Y), dim=1)
                ax.plot_surface(XX, YY, self.predict(X_input).reshape(XX.size()), color=model_color, alpha=model_opacity if model_opacity else 0.5, label=model_label)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                ax.set_zlabel(axis_labels[2])
            case _:
                print("Only linear models with 1 or 2 input parameters can be plotted")
                return
        plt.title(title)
        plt.legend()

    def plot_residuals(self):
        assert hasattr(self, "residuals"), "ElasticNetRegression.fit(x, y) must be called before plotting"
        fig, ax = plt.subplots(1, 2, figsize=(14,7))
        ax[0].plot(self.residuals, ".")
        ax[0].axhline(y=torch.mean(self.residuals))
        stats.probplot(self.residuals, dist="norm", plot=ax[1])
        ax[0].set_title('Residuals Plot')
        ax[0].set_xlabel('Index')
        ax[0].set_ylabel('Residuals')
        ax[1].set_title('Q-Q Plot')
    
    def _round_tuple(self, list, decimals=3):
        return tuple(round(item, decimals) for item in list)

    def summary(self, decimals=3):
        assert hasattr(self, "residuals"), "ElasticNetRegression.fit(x, y) must be called before getting a summary"
        print("======================== SUMMARY ========================")
        residual_quantiles = torch.min(self.residuals).item(), torch.quantile(self.residuals, 0.25).item(), torch.quantile(self.residuals, 0.50).item(), torch.quantile(self.residuals, 0.75).item(), torch.max(self.residuals).item()
        print(f"Residual quantiles: {self._round_tuple(residual_quantiles, decimals=decimals)}")
        print(f"Coefficients: {self._round_tuple(tuple(self.bias.tolist() + self.weights.tolist()), decimals=decimals)}")
        SSE = torch.sum(self.residuals ** 2).item()
        SST = torch.sum((self.Y - torch.mean(self.Y)) ** 2).item()
        r_squared = 1 - SSE / SST
        print(f"Coefficient of determination: {round(r_squared, decimals)}")
        n = self.X.shape[0]
        p = self.X.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        print(f"Adjusted R squared: {round(adjusted_r_squared, decimals)}")
