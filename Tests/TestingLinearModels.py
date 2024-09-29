import torch
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.SupervisedLearning.LinearModels.LinearRegression import LinearRegression
from src.DLL.MachineLearning.SupervisedLearning.LinearModels.RidgeRegression import RidgeRegression
from src.DLL.MachineLearning.SupervisedLearning.LinearModels.LassoRegression import LASSORegression


plt.style.use(["grid", "notebook"])

x = torch.linspace(0, 1, 20)
y = torch.linspace(0, 1, 20)
XX, YY = torch.meshgrid(x, y, indexing="xy")
X = XX.flatten()
Y = YY.flatten()
X_input = torch.stack((X, Y), dim=1)
Z = 2 * X - 5 * Y + torch.normal(0, 1, size=X.size())

model1 = LinearRegression()
model2 = RidgeRegression(alpha=0.1)
model3 = LASSORegression(alpha=0.0, learning_rate=0.01)
model1.fit(X_input, Z)
model1.summary()
model1.plot()
model2.fit(X_input, Z)
model2.summary()
model2.plot()
model3.fit(X_input, Z, epochs=1000)
model3.summary()
model3.plot()
plt.show()

model1.fit(torch.linspace(0, 1, 100), 2 * torch.linspace(0, 1, 100) + torch.normal(0, 0.1, size=(100,)))
model1.summary()
model1.plot()
model2.fit(torch.linspace(0, 1, 100), 2 * torch.linspace(0, 1, 100) + torch.normal(0, 0.1, size=(100,)))
model2.summary()
model2.plot()
model3.fit(torch.linspace(0, 1, 100), 2 * torch.linspace(0, 1, 100) + torch.normal(0, 0.1, size=(100,)), epochs=1000)
model3.summary()
model3.plot()
plt.show()
