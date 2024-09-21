import torch
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.SupervisedLearning.LinearModels.LinearRegression import LinearRegression


plt.style.use(["grid", "notebook"])

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
model.plot_residuals()
plt.show()

model.fit(torch.linspace(0, 1, 100), 2 * torch.linspace(0, 1, 100) + torch.normal(0, 0.1, size=(100,)))
model.summary()
model.plot()
model.plot_residuals()
plt.show()
