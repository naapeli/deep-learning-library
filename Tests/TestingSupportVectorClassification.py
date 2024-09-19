from src.DLL.MachineLearning.SupportVectorMachines.SVC import SVC
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import accuracy

import torch
import matplotlib.pyplot as plt
from sklearn import datasets


X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=4)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

X = torch.from_numpy(X)
y = torch.from_numpy(y) * 2 - 1
X_train, y_train, X_test, y_test, _, _ = data_split(X, y)

model = SVC()
model.fit(X_train, y_train, epochs=200)
predictions = model.predict(X_test)
print(accuracy(predictions, y_test))

# From AssemblyAI
def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=(y_test == predictions))

    x0_1 = torch.amin(X[:, 0])
    x0_2 = torch.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, model.weight, model.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, model.weight, model.bias, 0)

    x1_1_m = get_hyperplane_value(x0_1, model.weight, model.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.weight, model.bias, -1)

    x1_1_p = get_hyperplane_value(x0_1, model.weight, model.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.weight, model.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = torch.amin(X[:, 1])
    x1_max = torch.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

visualize_svm()
