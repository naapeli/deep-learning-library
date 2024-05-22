import torch
import matplotlib.pyplot as plt
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
        plt.style.use(["grid", "notebook"])
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
                ax.plot_surface(XX, YY, model.predict(X_input).reshape(XX.size()), color=model_color, alpha=model_opacity if model_opacity else 0.5, label=model_label)
                plt.xlabel(axis_labels[0])
                plt.ylabel(axis_labels[1])
                ax.set_zlabel(axis_labels[2])
            case _:
                print("Only linear models with 1 or 2 input parameters can be plotted")
                return
        plt.title(title)
        plt.legend()
        plt.show()
    
    def _round_tuple(self, list, decimals=3):
        return tuple(round(item, decimals) for item in list)

    def summary(self, decimals=3):
        assert hasattr(self, "residuals"), "LinearRegression.fit(x, y) must be called before getting a summary"
        print("======================== SUMMARY ========================")
        residual_quantiles = torch.min(self.residuals).item(), torch.quantile(self.residuals, 0.25).item(), torch.quantile(self.residuals, 0.50).item(), torch.quantile(self.residuals, 0.75).item(), torch.max(self.residuals).item()
        print(f"Residual quantiles: {self._round_tuple(residual_quantiles, decimals=decimals)}")
        print(f"Coefficients: {self._round_tuple(tuple(self.beta.tolist()), decimals=decimals)}")
        SSE = torch.sum(self.residuals ** 2)
        SST = torch.sum((self.Y - torch.mean(self.Y)) ** 2)
        r_squared = 1 - SSE / SST
        print(f"Coefficient of determination: {r_squared}")
        n = self.X.shape[0]
        p = self.X.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        print(f"Adjusted R squared: {adjusted_r_squared}")


# move testing code to test file!!!
x = torch.linspace(0, 1, 20)
y = torch.linspace(0, 1, 20)
XX, YY = torch.meshgrid(x, y, indexing="xy")
X = XX.flatten()
Y = YY.flatten()
X_input = torch.stack((X, Y), dim=1)
Z = 2 * X - 5 * Y + torch.normal(0, 1, size=X.size())

model = LinearRegression()
model.fit(X_input, Z)
model.summary()
model.plot()

model.fit(torch.linspace(0, 1, 100), 2 * torch.linspace(0, 1, 100) + torch.normal(0, 0.1, size=(100,)))
model.summary()
model.plot()


