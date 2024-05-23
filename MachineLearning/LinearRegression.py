import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import scienceplots


class LinearRegression:
    def fit(self, X, Y):
        if len(X.shape) == 1: X = X.unsqueeze(1)
        self.X = X
        self.Y = Y
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        self.beta = torch.inverse(X.T @ X) @ X.T @ Y
        self.residuals = self.Y - self.predict(self.X)

    def predict(self, X):
        assert hasattr(self, "beta"), "LinearRegression.fit(x, y) must be called before predicting"
        if len(X.shape) == 1: X = X.unsqueeze(1)
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        return X @ self.beta
    
    def plot(self, axis_labels=("x", "y", "z"), title="Linear regression", model_label="Model", scatter_label="Datapoints", model_color=None, scatter_color=None, model_opacity=None):
        assert hasattr(self, "beta"), "LinearRegression.fit(x, y) must be called before plotting"
        match self.X.shape[1]:
            case 1:
                x = self.X[:, 0]
                plt.plot(x, self.Y, ".", color=scatter_color, label=scatter_label)
                plt.plot(x, self.predict(self.X), color=model_color, alpha=model_opacity, label=model_label)
                plt.xlabel(axis_labels[0])
                plt.ylabel(axis_labels[1])
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
                plt.xlabel(axis_labels[0])
                plt.ylabel(axis_labels[1])
                ax.set_zlabel(axis_labels[2])
            case _:
                print("Only linear models with 1 or 2 input parameters can be plotted")
                return
        plt.title(title)
        plt.legend()

    def plot_residuals(self):
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
        assert hasattr(self, "residuals"), "LinearRegression.fit(x, y) must be called before getting a summary"
        print("======================== SUMMARY ========================")
        residual_quantiles = torch.min(self.residuals).item(), torch.quantile(self.residuals, 0.25).item(), torch.quantile(self.residuals, 0.50).item(), torch.quantile(self.residuals, 0.75).item(), torch.max(self.residuals).item()
        print(f"Residual quantiles: {self._round_tuple(residual_quantiles, decimals=decimals)}")
        print(f"Coefficients: {self._round_tuple(tuple(self.beta.tolist()), decimals=decimals)}")
        SSE = torch.sum(self.residuals ** 2).item()
        SST = torch.sum((self.Y - torch.mean(self.Y)) ** 2).item()
        r_squared = 1 - SSE / SST
        print(f"Coefficient of determination: {round(r_squared, decimals)}")
        n = self.X.shape[0]
        p = self.X.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        print(f"Adjusted R squared: {round(adjusted_r_squared, decimals)}")
