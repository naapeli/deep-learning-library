import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as gbr

from src.DLL.MachineLearning.SupervisedLearning.RandomForests import RegressionTree
from src.DLL.MachineLearning.SupervisedLearning.RandomForests import RandomForestRegressor
from src.DLL.MachineLearning.SupervisedLearning.RandomForests import GradientBoostingRegressor


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

model3 = GradientBoostingRegressor(n_trees=10, learning_rate=0.5)
model3.fit(x,y)
y_pred3 = model3.predict(x_test)

model4 = gbr(n_estimators=10, learning_rate=0.5)
model4.fit(x,y)
y_pred4 = model4.predict(x_test)

plt.plot(x.numpy(), y.numpy(), color="Blue")
plt.plot(x_test.numpy(), y_pred.numpy(), color="Red")
plt.plot(x_test.numpy(), y_pred2.numpy(), color="Green")
plt.plot(x_test.numpy(), y_pred3.numpy(), color="Yellow")
plt.plot(x_test.numpy(), y_pred4, color="gray")
plt.show()
