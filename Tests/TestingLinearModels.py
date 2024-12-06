import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
import scienceplots

from src.DLL.MachineLearning.SupervisedLearning.LinearModels import LinearRegression, RidgeRegression, LASSORegression, ElasticNetRegression
from src.DLL.Data.Metrics import r2_score, adjusted_r2_score
from src.DLL.DeepLearning.Optimisers import LBFGS, ADAM


plt.style.use(["grid", "notebook"])

def summary(predictions, true_values, n_features):
    print("======================== SUMMARY ========================")
    residuals = true_values - predictions
    residual_quantiles = torch.min(residuals).item(), torch.quantile(residuals, 0.25).item(), torch.quantile(residuals, 0.50).item(), torch.quantile(residuals, 0.75).item(), torch.max(residuals).item()
    print(f"Residual quantiles: {tuple(round(item, 3) for item in residual_quantiles)}")
    r_squared = r2_score(predictions, true_values)
    print(f"Coefficient of determination: {round(r_squared, 3)}")
    adjusted_r_squared = adjusted_r2_score(predictions, true_values, n_features)
    print(f"Adjusted R squared: {round(adjusted_r_squared, 3)}")

def plot_residuals(predictions, true_values):
    fig, ax = plt.subplots(1, 2, figsize=(14,7))
    residuals = true_values - predictions
    ax[0].plot(residuals, ".")
    ax[0].axhline(y=torch.mean(residuals))
    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[0].set_title('Residuals Plot')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Residuals')
    ax[1].set_title('Q-Q Plot')

def plot1d(x, true_values, predictions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, true_values, ".", color="red", label="true values")
    ax.plot(x, predictions, color="blue", label="predictions")
    ax.legend()
    ax.set_title(title)

def plot2d(model, X, true_values, title):
    x = X[:, 0]
    y = X[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, true_values, label="true values", color="red")
    x = torch.linspace(torch.min(x), torch.max(x), 2)
    y = torch.linspace(torch.min(y), torch.max(y), 2)
    XX, YY = torch.meshgrid(x, y, indexing="xy")
    X = XX.flatten()
    Y = YY.flatten()
    X_input = torch.stack((X, Y), dim=1)
    ax.plot_surface(XX, YY, model.predict(X_input).reshape(XX.size()), color="blue", alpha=0.5, label="predictions")
    ax.legend()
    ax.set_title(title)


x = torch.linspace(0, 1, 20)
y = torch.linspace(0, 1, 20)
XX, YY = torch.meshgrid(x, y, indexing="xy")
X = XX.flatten()
Y = YY.flatten()
X_input = torch.stack((X, Y), dim=1)
Z = 2 * X - 5 * Y + torch.normal(0, 1, size=X.size())

model1 = LinearRegression()
model2 = RidgeRegression(alpha=1.0)
model3 = LASSORegression(alpha=0.1)
model4 = ElasticNetRegression(alpha=0.1, l1_ratio=0.2)
model1.fit(X_input, Z, method="ols")
summary(model1.predict(X_input), Z, X_input.shape[1])
plot2d(model1, X_input, Z, "Linear regression")
plot_residuals(model1.predict(X_input), Z)
model2.fit(X_input, Z)
summary(model2.predict(X_input), Z, X_input.shape[1])
plot2d(model2, X_input, Z, "Ridge regression")
model3.fit(X_input, Z, epochs=100, optimiser=LBFGS(lambda: model3.loss.loss(model3.predict(X_input), Z), learning_rate=0.1))
summary(model3.predict(X_input), Z, X_input.shape[1])
plot2d(model3, X_input, Z, "LASSO regression")
model4.fit(X_input, Z, epochs=1000, optimiser=ADAM(learning_rate=0.1))
summary(model4.predict(X_input), Z, X_input.shape[1])
plot2d(model4, X_input, Z, "Elasticnet regression")
plt.show()

X = torch.linspace(0, 1, 100).unsqueeze(dim=1)
model1.fit(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), method="tls")
summary(model1.predict(X), 2 * X.squeeze(), 1)
plot1d(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), model1.predict(X), "Linear regression")
model2.fit(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)))
summary(model2.predict(X), 2 * X.squeeze(), 1)
plot1d(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), model2.predict(X), "Ridge regression")
model3.fit(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), epochs=1000, optimiser=ADAM(learning_rate=0.1))
summary(model3.predict(X), 2 * X.squeeze(), 1)
plot1d(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), model3.predict(X), "LASSO regression")
history = model4.fit(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), epochs=10, metrics=["rmse"], optimiser=LBFGS(lambda: model4.loss.loss(model4.predict(X), 2 * X.squeeze()), learning_rate=0.01))
summary(model4.predict(X), 2 * X.squeeze(), 1)
plot1d(X, 2 * X.squeeze() + torch.normal(0, 0.1, size=(100,)), model4.predict(X), "Elasticnet regression")

plt.figure()
plt.plot(history["rmse"])
plt.title("Elasticnet regresson")
plt.xlabel("epoch")
plt.ylabel("rmse")

plt.show()
