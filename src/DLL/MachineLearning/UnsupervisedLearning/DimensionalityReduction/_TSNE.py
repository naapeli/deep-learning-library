import torch

from . import PCA
from ....DeepLearning.Optimisers import SGD


class TSNE:
    """
    T-distributed Stochastic Neighbor Embedding (T-SNE) class for dimensionality reduction. This implementation is based on `this paper <https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`_ and `this article <https://towardsdatascience.com/t-sne-from-scratch-ft-numpy-172ee2a61df7/>`_. The main difference is that this implementation uses vectorized matrix operations making it considerably faster than the loop approach used in the article.

    Args:
        n_components (int): Number of principal components to keep. The number must be a positive integer.
        init (str, optional): The method for initializing the embedding. Must be in ``["pca", "random"]``. Defaults to ``"pca"``.
        p (int, optional): The order of the chosen metric. Must be a positive integer. Defaults to 2, which corresponds to the Euclidian metric.
        early_exaggeration (float | int, optional): Determines how far apart the clusters are in the embedding space. Must be a positive real number. Defaults to 12.0.
        perplexity (float | int, optional): Determines how far can samples be from one another to be considered neighbors. Must be a positive real number. Defaults to 30.0. One should consider using something between 5 and 50 to begin with.
        learning_rate (float | int, optional): Determines how long steps do we take towards the gradient. Must be a positive real number. It is recommended to use a value between 10.0 and 1000.0. Defaults to ``"auto"``, where we use a value of ``max(n_samples / (4 * early_exaggeration), 50)``.
    
    Attributes:
        history (list[float]): The history of KL-divergence loss function each epoch. Available after fitting the model.
    """
    def __init__(self, n_components=2, init="pca", p=2, early_exaggeration=12.0, perplexity=30.0, learning_rate="auto"):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not init in ["pca", "random"]:
            raise ValueError('init must be in ["pca", "random"].')
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p must be a positive integer.")
        if not isinstance(early_exaggeration, float | int) or early_exaggeration <= 0:
            raise ValueError("early_exaggeration must be a positive real number.")
        if not isinstance(perplexity, float | int) or perplexity <= 0:
            raise ValueError("perplexity must be a positive real number.")
        if (learning_rate != "auto" and (not isinstance(learning_rate, float | int) or learning_rate <= 0)):
            raise ValueError('learning_rate must be a positive real number or "auto".')

        self.n_components = n_components
        self.init = init
        self.p = p
        self.early_exaggeration = early_exaggeration
        self.perplexity = perplexity
        self.learning_rate = learning_rate
    
    def _initialize(self, X):
        if self.init == "pca":
            return PCA(n_components=self.n_components).fit_transform(X)
        else:
            return torch.normal(0, 1e-4, size=(len(X), self.n_components))
    
    def _symmetric_affinities(self, X):
        affinities = self._pairwise_affinities(X)
        p_ij_symmetric = (affinities + affinities.T) / (2 * len(affinities))
        eps = torch.tensor(torch.finfo(p_ij_symmetric.dtype).tiny)
        p_ij_symmetric = torch.maximum(p_ij_symmetric, eps)
        return p_ij_symmetric
    
    def _pairwise_affinities(self, X):
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        norm = torch.norm(diff, dim=2, p=self.p)
        variance = self._grid_search(norm)
        affinities = torch.exp(-norm ** self.p / (2 * variance[:, None] ** 2))
        affinities.fill_diagonal_(0)
        row_sums = affinities.sum(dim=1, keepdim=True)
        zero_rows = row_sums == 0
        eps = torch.finfo(affinities.dtype).tiny
        row_sums[zero_rows] = eps
        affinities /= row_sums
        affinities[affinities == 0] = eps
        return affinities

    def _grid_search(self, norm):
        n_samples = len(norm)
        std_norm = torch.std(norm, dim=1, keepdim=True)
        resolution = 200
        sigma_search_values = torch.linspace(0.01, 5.0, resolution)
        sigma_search_values = std_norm * sigma_search_values
        p_matrix = torch.exp(-norm.unsqueeze(2) ** self.p / (2 * sigma_search_values.pow(2).unsqueeze(1)))
        eps = torch.tensor(torch.finfo(p_matrix.dtype).tiny)
        p_matrix[torch.eye(n_samples, dtype=torch.bool)] = eps  # do not set to 0 to avoid division by zero errors
        p_new_matrix = torch.maximum(p_matrix / p_matrix.sum(dim=1, keepdim=True), eps)
        H_values = -torch.sum(p_new_matrix * torch.log2(p_new_matrix), dim=1)
        result = torch.abs(torch.log(torch.tensor(self.perplexity)) - H_values * torch.log(torch.tensor(2.0)))
        sigma_indices = torch.argmin(result, dim=1)
        return sigma_search_values[torch.arange(n_samples), sigma_indices]

    def _low_dimensional_affinities(self, low_dimensional_representation):
        diff = low_dimensional_representation.unsqueeze(1) - low_dimensional_representation.unsqueeze(0)
        norm = torch.norm(diff, dim=2, p=self.p)
        result = (1 + norm ** self.p).pow(-1)
        result.fill_diagonal_(0)
        result_sum = result.sum()
        result = result / result_sum if result_sum > 0 else result
        eps = torch.tensor(torch.finfo(result.dtype).tiny)
        return torch.maximum(result, eps)

    def _gradient(self, affinities, low_dimensional_affinities, low_dimensional_representation):
        diff = low_dimensional_representation.unsqueeze(1) - low_dimensional_representation.unsqueeze(0)
        A = affinities - low_dimensional_affinities
        B = (1 + torch.norm(diff, dim=2, p=self.p)).pow(-1)
        return 4 * torch.sum(((A * B).unsqueeze(2) * diff), dim=1)

    def fit_transform(self, X, epochs=100, verbose=False):
        """
        Fits the T-SNE model to the input data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            epochs (int, optional): The number of training epochs after early exaggeration. Must be a positive integer. Defaults to 100. Due to early exaggeration, the embedding is updated epochs + 250 times.
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
        learning_rate = max(len(X) / (4 * self.early_exaggeration), 50) if self.learning_rate == "auto" else self.learning_rate
        optimiser = SGD(learning_rate=learning_rate, momentum=0.5)
        optimiser.initialise_parameters([result])
        self.history = []
        early_exaggeration = self.early_exaggeration

        for epoch in range(epochs + 250):
            if epoch > 250:
                optimiser.momentum = 0.8
                early_exaggeration = 1
            low_dimensional_affinities = self._low_dimensional_affinities(result)
            result.grad = self._gradient(early_exaggeration * symmetric_affinities, low_dimensional_affinities, result)
            loss = torch.sum(symmetric_affinities * torch.log(symmetric_affinities / low_dimensional_affinities))
            self.history.append(loss)
            if verbose: print(f"Epoch: {epoch + 1} - KL-divergence loss: {loss:.3f}")
            optimiser.update_parameters()

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
