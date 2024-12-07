import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA as PCA_sklearn

from src.DLL.MachineLearning.UnsupervisedLearning.DimensionalityReduction import PCA, LDA


# (images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
# X = torch.from_numpy(images).to(dtype=torch.float64).reshape(60000, -1)
# y = torch.from_numpy(labels).to(dtype=torch.int32)
iris = datasets.load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.float32)

transformer_pca = PCA(n_components=2)
reduced_pca = transformer_pca.fit_transform(X, normalize=False)

sk_pca = PCA_sklearn(n_components=2).fit_transform(X)
# plt.plot(transformer.explained_variance, ".")

transformer_lda = LDA(n_components=2)
reduced_lda = transformer_lda.fit_transform(X, y)

fig, axes = plt.subplots(1, 3)
axes = axes.ravel()
axes[0].scatter(reduced_pca[:, 0], reduced_pca[:, 1], c=y, s=5)
axes[0].set_title("Own implementation PCA")
axes[1].scatter(sk_pca[:, 0], sk_pca[:, 1], c=y, s=5)
axes[1].set_title("SKlearn implementation PCA")
axes[2].scatter(reduced_lda[:, 0], reduced_lda[:, 1], c=y, s=5)
axes[2].set_title("Own implementation LDA")
plt.show()
