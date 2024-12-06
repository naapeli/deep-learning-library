import torch

from ...SupervisedLearning.Kernels import RBF, _Base
from . import KMeansClustering
from ....Exceptions import NotFittedError


class SpectralClustering:
    """
    SpectralClustering implements the spectral clustering algorithm, which partitions data points into `k` clusters.

    Args:
        kernel (:ref:`kernel_section_label`, optional): The similarity function for fitting the model. Defaults to RBF(correlation_length=0.1).
        k (int, optional): The number of clusters. Defaults to 3. Must be a positive integer.
        max_iters (int, optional): The maximum number of iterations for training the model. Defaults to 100. Must be a positive integer.
        normalise (bool, optional): Determines if the laplacian matrix is calculated using L = I - sqrt(inv(D)) A sqrt(inv(D)) or just L = D - A. Defaults to True.
        use_kmeans (bool, optional): Determines if the clustring in embedded space is done using kmeans or discretisation. Defaults to True.
        **kwargs: Other arguments are passed into the `KMeansClustering` algorithm.

    Note:
        The result depends heavily on the chosen kernel function. Especially the correlation_length parameter should be fine-tuned for optimal performance.
    """
    def __init__(self, kernel=RBF(correlation_length=0.1), k=3, max_iters=100, normalise=True, use_kmeans=True, **kwargs):
        if not isinstance(kernel, _Base):
            raise ValueError("kernel must be from DLL.MachineLearning.SupervisedLearning.Kernels")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(max_iters, int) or max_iters < 1:
            raise ValueError("max_iters must be a positive integer.")
        if not isinstance(normalise, bool):
            raise TypeError("normalise must be a boolean.")
        if not isinstance(use_kmeans, bool):
            raise TypeError("use_kmeans must be a boolean.")

        self.kernel = kernel
        self.k = k
        self.max_iters = max_iters
        self.normalise = normalise
        self.use_kmeans = use_kmeans
        if self.use_kmeans: self.kmeans = KMeansClustering(k=self.k, max_iters=self.max_iters, **kwargs)
    
    def _transform_data(self, X):
        similarity_matrix = self.kernel(X, X)
        if self.normalise:
            D_sqrt_inv = torch.diag(1.0 / torch.sqrt(similarity_matrix.sum(axis=1)))
            laplacian_matrix = torch.eye(X.shape[0]) - D_sqrt_inv @ similarity_matrix @ D_sqrt_inv
        else:
            degree_matrix = torch.diag(similarity_matrix.sum(axis=1))
            laplacian_matrix = degree_matrix - similarity_matrix
        
        _, eigenvectors = torch.linalg.eigh(laplacian_matrix)
        selected_eigenvectors = eigenvectors[:, :self.k]
        return selected_eigenvectors
    
    def _discretize_spectral_embedding(self, selected_eigenvectors):
        selected_eigenvectors = selected_eigenvectors / torch.linalg.norm(selected_eigenvectors, axis=1, keepdims=True)
        n_samples = len(selected_eigenvectors)
        H = torch.zeros((n_samples, self.k), dtype=selected_eigenvectors.dtype)
        indices = torch.argmax(selected_eigenvectors, axis=1)
        H[torch.arange(n_samples), indices] = 1

        for _ in range(self.max_iters):
            U, _, Vt = torch.linalg.svd(selected_eigenvectors.T @ H)
            Q = U @ Vt
            H_new = torch.argmax(selected_eigenvectors @ Q, axis=1)
            H = torch.zeros((n_samples, self.k), dtype=selected_eigenvectors.dtype)
            H[torch.arange(n_samples), H_new] = 1
        
        labels = torch.argmax(H, axis=1)
        return labels
    
    def fit(self, X):
        """
        Fits the algorithm to the given data. Transforms the data into the embedding space using the kernel function and clusters the data in the space.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature. The number of samples must be atleast `k`.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] < self.k:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast k samples.")
        
        self.selected_eigenvectors = self._transform_data(X)
        if self.use_kmeans:
            self.kmeans.fit(self.selected_eigenvectors)
        else:
            self._labels = self._discretize_spectral_embedding(self.selected_eigenvectors)
    
    def predict(self):
        """
        Applies the fitted SpectralClustering model to the input data, partitioning it to `k` clusters.

        Returns:
            labels (torch.Tensor of shape (n_samples,)): The cluster corresponding to each sample.
        Raises:
            NotFittedError: If the SpectralClustering model has not been fitted before predicting.
        """
        if not hasattr(self, "selected_eigenvectors"):
            raise NotFittedError("SpectralClustering.fit() must be called before predicting.")
        
        if self.use_kmeans:
            return self.kmeans.predict(self.selected_eigenvectors)
        else:
            return self._labels
