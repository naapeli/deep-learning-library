import torch
from scipy.optimize import curve_fit
from functools import partial

from . import PCA
from ....DeepLearning.Optimisers import SGD
from ...SupervisedLearning.Kernels import RBF


class UMAP:
    """
    Uniform Manifold Approximation and Projection (UMAP) class for dimensionality reduction. This implementation is based on `this paper <https://arxiv.org/pdf/1802.03426>`_ and `this article <https://github.com/NikolayOskolkov/HowUMAPWorks/blob/master/HowUMAPWorks.ipynb>`_.

    Args:
        n_components (int): Number of principal components to keep. The number must be a positive integer.
        init (str, optional): The method for initializing the embedding. Must be in ``["spectral", "pca", "random"]``. Defaults to ``"spectral"``.
        p (int, optional): The order of the chosen metric. Must be a positive integer. Defaults to 2, which corresponds to the Euclidian metric.
        n_neighbor (int, optional): Controls how UMAP balances local and global structure in data. The larger this parameter is the better the global structure is conserved. A small value conserves fine details well, but may lose global structure. Must be a positive integer. Defaults to 15.
        min_dist (float | int, optional): Controls the minimum distance between samples in the low dimensional space. Must be a non-negative real number. Defaults to 0.25.
        learning_rate (float | int, optional): Determines how long steps do we take towards the gradient. Must be a positive real number. Defaults to 1.

    Attributes:
        history (list[float]): The history of the cross entropy loss function each epoch. Available after fitting the model.
    """
    def __init__(self, n_components=2, init="spectral", p=2, n_neighbor=15, min_dist=0.25, learning_rate=1):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not init in ["spectral", "pca", "random"]:
            raise ValueError('init must be in ["spectral", "pca", "random"].')
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p must be a positive integer.")
        if not isinstance(n_neighbor, int) or n_neighbor <= 0:
            raise ValueError("n_neighbor must be a positive integer.")
        if not isinstance(min_dist, int | float) or min_dist < 0:
            raise ValueError("min_dist must be a non-negative real number.")
        if (not isinstance(learning_rate, float | int) or learning_rate <= 0):
            raise ValueError('learning_rate must be a positive real number.')

        self.n_components = n_components
        self.init = init
        self.p = p
        self.learning_rate = learning_rate
        self.n_neighbor = n_neighbor
        self.min_dist = min_dist
        self.a, self.b = self._solve_params()

    def _solve_params(self):
        def func(d, a, b):
            return 1 / (1 + a * d ** (2 * b))
        
        x = torch.linspace(0, 3, 300)
        y = torch.full_like(x, 1)
        mask = x > self.min_dist
        y[mask] = torch.exp(-x[mask] + self.min_dist)
        p, _ = curve_fit(func, x.numpy(), y.numpy())
        return torch.tensor(p[0], dtype=torch.float32), torch.tensor(p[1], dtype=torch.float32)  # a, b

    def _initialize(self, X):
        if self.init == "pca":
            return PCA(n_components=self.n_components).fit_transform(X)
        elif self.init == "spectral":
            W = RBF()(X, X)
            D = torch.diag(W.sum(dim=1))
            L = D - W
            _, eigenvectors = torch.linalg.eigh(L)
            embedding = eigenvectors[:, 1:self.n_components+1]
            embedding /= torch.linalg.norm(embedding, dim=1, keepdim=True) + 1e-10
            return embedding
        else:
            return torch.normal(0, 1e-4, size=(len(X), self.n_components))

    def _pairwise_affinities(self, norm, rho, sigma, row):
        d = norm[row] - rho[row]
        d[d < 0] = 0
        return torch.exp(- d / sigma)

    def _k(self, prob):
        return 2 ** torch.sum(prob)

    def _sigma(self, k_of_sigma, fixed_k):
        sigma_lower_limit = 0
        sigma_upper_limit = 1000
        for _ in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if k_of_sigma(approx_sigma) < fixed_k:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if torch.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def _local_fuzzy_simplicial_set(self, norm, rho):
        n = len(norm)
        prob = torch.zeros((n, n))
        for dist_row in range(n):
            _func = partial(self._k_of_sigma, norm, rho, dist_row)
            prob[dist_row] = self._pairwise_affinities(norm, rho, self._sigma(_func, self.n_neighbor), dist_row)
        return prob
    
    def _k_of_sigma(self, norm, rho, dist_row, sigma):
        return self._k(self._pairwise_affinities(norm, rho, sigma, dist_row))

    def _symmetric_affinities(self, X):
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        norm = torch.norm(diff, dim=2, p=self.p) ** 2
        rho = [sorted(norm[i])[1] for i in range(len(norm))]  # distance to nearest neighbor
        affinities = self._local_fuzzy_simplicial_set(norm, rho)
        return (affinities + affinities.T) / 2  # suggested by article
        # return affinities + affinities.T - affinities * affinities.T  # suggested by paper

    def _low_dimensional_affinities(self, low_dimensional_representation):
        diff = low_dimensional_representation.unsqueeze(1) - low_dimensional_representation.unsqueeze(0)
        norm = torch.norm(diff, dim=2, p=self.p)
        inv_distances = 1 / (1 + self.a * norm ** (2 * self.b))
        return inv_distances

    def _CE(self, symmetric_affinities, low_dimensional_representation):
        low_dimensional_affinities = self._low_dimensional_affinities(low_dimensional_representation)
        return torch.mean(-symmetric_affinities * torch.log(low_dimensional_affinities + 1e-5) - (1 - symmetric_affinities) * torch.log(1 - low_dimensional_affinities + 1e-5))

    def _CE_gradient(self, symmetric_affinities, low_dimensional_representation):
        diff = low_dimensional_representation.unsqueeze(1) - low_dimensional_representation.unsqueeze(0)
        norm = torch.norm(diff, dim=2, p=self.p) ** 2
        inv_dist = 1 / (1 + self.a * norm ** self.b)
        Q = (1 - symmetric_affinities) @ (1 / (norm + 1e-5))
        Q.fill_diagonal_(0)
        Q /= torch.sum(Q, dim=1, keepdims=True)
        fact = (self.a * symmetric_affinities * (1e-8 + norm) ** (self.b - 1) - Q).unsqueeze(2)
        return 2 * self.b * torch.sum(fact * diff * inv_dist.unsqueeze(2), dim=1)

    def fit_transform(self, X, epochs=100, verbose=False):
        """
        Fits the UMAP model to the input data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            epochs (int, optional): The number of training epochs. Must be a positive integer. Defaults to 100.
            verbose (bool, optional): Determines if the loss is printed each epoch. Must be a boolean. Defaults to ``False``.
        Returns:
            embedding (torch.tensor): The embedded samples of shape (n_samples, n_components).
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] == 1:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast 2 samples.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean.")

        result = self._initialize(X)
        symmetric_affinities = self._symmetric_affinities(X)
        optimiser = SGD(learning_rate=self.learning_rate, momentum=0.5)
        optimiser.initialise_parameters([result])
        self.history = []

        for epoch in range(epochs):
            result.grad = self._CE_gradient(symmetric_affinities, result)
            optimiser.update_parameters()
            loss = self._CE(symmetric_affinities, result)
            self.history.append(loss)
            if verbose: print(f"Epoch: {epoch + 1} - Cross-Entropy loss: {loss:.3f}")

        return result

    def fit(self, X, epochs=100, verbose=False):
        """
        Wrapper for the TSNE.fit_transform(X) method.
        
        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            epochs (int, optional): The number of training epochs after early exaggeration. Must be a positive integer. Defaults to 100.
            verbose (bool, optional): Determines if the loss is printed each epoch. Must be a boolean. Defaults to ``False``.
        """
        return self.fit_transform(X, epochs, verbose)
