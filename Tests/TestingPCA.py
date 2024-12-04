import torch
# import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA as PCA_sklearn

from src.DLL.MachineLearning.UnsupervisedLearning.DimensionalityReduction import PCA


# (images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
# X = torch.from_numpy(images).to(dtype=torch.float64).reshape(60000, -1)
# y = torch.from_numpy(labels).to(dtype=torch.int32)
iris = datasets.load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.float32)

transformer = PCA(n_components=2)
transformer.fit(X, normalize=False)
reduced = transformer.transform(X)

# plt.plot(transformer.explained_variance, ".")

sk_reduced = PCA_sklearn(n_components=2).fit_transform(X)

fig, axes = plt.subplots(1, 2)
axes[0].scatter(reduced[:, 0], reduced[:, 1], c=y, s=5)
axes[0].set_title("Own implementation")
axes[1].scatter(sk_reduced[:, 0], sk_reduced[:, 1], c=y, s=5)
axes[1].set_title("SKlearn implementation")
plt.show()
