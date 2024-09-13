import torch
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.RandomForests.RegressionTree import RegressionTree
from src.DLL.MachineLearning.RandomForests.RegressionForest import RandomForestRegressor


n = 100
x = torch.linspace(0, 1, n).unsqueeze(-1)
y = x * x + torch.normal(mean=0, std=0.05, size=(n, 1))

model = RegressionTree()
model.fit(x, y)
x_test, _ = torch.rand_like(x).sort(dim=0)
y_pred = model.predict(x_test)

model2 = RandomForestRegressor(n_trees=10)
model2.fit(x, y)
y_pred2 = model2.predict(x_test)


plt.plot(x.numpy(), y.numpy(), color="Blue")
plt.plot(x_test.numpy(), y_pred.numpy(), color="Red")
plt.plot(x_test.numpy(), y_pred2.numpy(), color="Green")
plt.show()
