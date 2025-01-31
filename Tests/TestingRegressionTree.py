import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as gbr, AdaBoostRegressor as abr
from sklearn.tree import DecisionTreeRegressor
from time import perf_counter

from src.DLL.MachineLearning.SupervisedLearning.Trees import RegressionTree, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, XGBoostingRegressor, LGBMRegressor
from src.DLL.Data.Preprocessing import data_split


n = 100
x = torch.linspace(0, 1, n).unsqueeze(-1)
y = 0.2 * torch.sin(20 * x) + x * x + torch.normal(mean=0, std=0.05, size=(n, 1))
y = y.squeeze()
rand_feats = torch.randint_like(x, 2)
x = torch.cat([x, rand_feats], dim=1)

model = RegressionTree()
model.fit(x, y)
x_test, _ = torch.rand((n, 1)).sort(dim=0)
x_test_rand_feats = torch.randint_like(x_test, 2)
x_test = torch.cat([x_test, x_test_rand_feats], dim=1)
y_pred = model.predict(x_test)

model2 = RandomForestRegressor(n_trees=3)
model2.fit(x, y)
y_pred2 = model2.predict(x_test)

model3 = GradientBoostingRegressor(n_trees=50, learning_rate=0.05, loss="absolute", max_depth=3)
history = model3.fit(x, y, metrics=["loss"])
y_pred3 = model3.predict(x_test)
plt.figure()
plt.plot(history["loss"])
plt.ylabel("Loss")
plt.xlabel("Tree")
plt.title("Gradient boosting regressor loss as a function of fitted trees")

model4 = gbr(n_estimators=10, learning_rate=0.5, loss="absolute_error")
model4.fit(x, y.ravel())
y_pred4 = model4.predict(x_test)

loss_adaboost = "exponential"
model5 = AdaBoostRegressor(n_trees=50, loss=loss_adaboost, max_depth=3)
errors = model5.fit(x, y)
y_pred5 = model5.predict(x_test)
plt.figure()
plt.title("Ada boost errors and confidences")
plt.plot(errors, label="errors")
plt.plot(model5.confidences, label="confidences")
plt.legend()

model6 = abr(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, loss=loss_adaboost)
model6.fit(x, y.ravel())
y_pred6 = model6.predict(x_test)

model7 = XGBoostingRegressor(n_trees=50, learning_rate=0.2, loss="huber", max_depth=3, reg_lambda=0.01, gamma=0, huber_delta=5)
start = perf_counter()
history = model7.fit(x, y, metrics=["loss"])
print(f"XGBoost time with {model7.n_trees} weak learners: {perf_counter() - start}")
y_pred7 = model7.predict(x_test)
plt.figure()
plt.plot(history["loss"])
plt.ylabel("Loss")
plt.xlabel("Tree")
plt.title("Extreme gradient boosting regressor loss as a function of fitted trees")

model8 = LGBMRegressor(n_trees=150, learning_rate=0.2, loss="squared", max_depth=3, reg_lambda=0.01, gamma=0, huber_delta=5, large_error_proportion=0.3, small_error_proportion=0.2)
start = perf_counter()
history = model8.fit(x, y, metrics=["loss"])
print(f"LGBM time with {model8.n_trees} weak learners: {perf_counter() - start}")
y_pred8 = model8.predict(x_test)
plt.figure()
plt.plot(history["loss"])
plt.ylabel("Loss")
plt.xlabel("Tree")
plt.title("Light gradient boosting machine regressor loss as a function of fitted trees")

plt.figure()
plt.plot(x[:, 0].numpy(), y.numpy(), color="Blue")
plt.plot(x_test[:, 0].numpy(), y_pred.numpy(), color="Red")
plt.plot(x_test[:, 0].numpy(), y_pred2.numpy(), color="Green")
plt.plot(x_test[:, 0].numpy(), y_pred3.numpy(), color="Yellow")
plt.plot(x_test[:, 0].numpy(), y_pred4, color="gray")
plt.plot(x_test[:, 0].numpy(), y_pred5.numpy(), color="brown")
plt.plot(x_test[:, 0].numpy(), y_pred6, color="pink")
plt.plot(x_test[:, 0].numpy(), y_pred7.numpy(), color="lightblue")
plt.plot(x_test[:, 0].numpy(), y_pred8.numpy(), color="black")
plt.show()

n = 20
X, Y = torch.meshgrid(torch.linspace(-1, 1, n, dtype=torch.float32), torch.linspace(-1, 1, n, dtype=torch.float32), indexing="xy")
x = torch.stack((X.flatten(), Y.flatten()), dim=1)
y = X.flatten() ** 2 + Y.flatten() ** 2 + 0.1 * torch.randn(size=Y.flatten().size()) - 5
x_train, y_train, _, _, x_test, y_test = data_split(x, y, train_split=0.8, validation_split=0.0)

model.fit(x_train, y_train)
z1 = model.predict(x_test)

model2.fit(x_train, y_train)
z2 = model2.predict(x_test)

history = model3.fit(x_train, y_train)
z3 = model3.predict(x_test)
plt.figure()
plt.plot(history["loss"])
plt.ylabel("Loss")
plt.xlabel("Tree")
plt.title("Gradient boosting regressor loss as a function of fitted trees")

model4.fit(x_train, y_train)
z4 = model4.predict(x_test)

errors = model5.fit(x_train, y_train)
z5 = model5.predict(x_test)
plt.figure()
plt.title("Ada boost errors and confidences")
plt.plot(errors, label="errors")
plt.plot(model5.confidences, label="confidences")
plt.legend()

model6.fit(x_train, y_train)
z6 = model6.predict(x_test)

history = model7.fit(x_train, y_train)
z7 = model7.predict(x_test)
plt.figure()
plt.plot(history["loss"])
plt.ylabel("Loss")
plt.xlabel("Tree")
plt.title("Extreme gradient boosting regressor loss as a function of fitted trees")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color="Blue")
ax.scatter(x_test[:, 0], x_test[:, 1], z1, color="Red")
ax.scatter(x_test[:, 0], x_test[:, 1], z2, color="Green")
ax.scatter(x_test[:, 0], x_test[:, 1], z3, color="Yellow")
ax.scatter(x_test[:, 0], x_test[:, 1], z4, color="gray")
ax.scatter(x_test[:, 0], x_test[:, 1], z5, color="brown")
ax.scatter(x_test[:, 0], x_test[:, 1], z6, color="pink")
ax.scatter(x_test[:, 0], x_test[:, 1], z7, color="lightblue")
plt.show()
