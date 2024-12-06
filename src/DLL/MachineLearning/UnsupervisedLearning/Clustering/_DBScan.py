import torch
from scipy.spatial import KDTree

from ....Exceptions import NotFittedError


class DBScan:
    """
    Density based scanning algorithm for clustering.

    Args:
        eps (float | int, optional): The distance inside of which datapoints are considered to be neighbours. Must be a positive real number. Defaults to 0.5.
        min_samples (int, optional): The minimum number of neighbours for a non-leaf node. Must be a positive integer. Defaults to 5.
    
    Note:
        The algorithm is very sensitive to changes in `eps`. One should fine-tune the value of eps for optimal results.
    """
    def __init__(self, eps=0.5, min_samples=5):
        if not isinstance(eps, int | float) or eps <= 0:
            raise ValueError("eps must be a positive real number.")
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer.")
        
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """
        Fits the algorithm to the given data. Recursively finds clusters by connecting near-by samples into the same cluster.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        
        X = X.numpy()
        tree = KDTree(X)
        n = len(X)
        labels = torch.zeros(n)
        current_class = 0

        # loop over samples to find roots for clusters.
        for i in range(n):
            if labels[i] != 0:
                continue
            
            neighbour_indicies = tree.query_ball_point(X[i], r=self.eps)
            
            if len(neighbour_indicies) < self.min_samples:
                labels[i] = -1
            else:
                current_class += 1
                labels[i] = current_class
                self._grow_cluster(X, tree, labels, neighbour_indicies, current_class)
        labels[labels != -1] -= 1  # transform the classes to start from 0.
        self._labels = labels

    def _grow_cluster(self, X, tree, labels, neighbour_indicies, current_class):
        j = 0
        while j < len(neighbour_indicies):
            neighbour_index = neighbour_indicies[j]
            
            if labels[neighbour_index] == -1:
                labels[neighbour_index] = current_class
            elif labels[neighbour_index] == 0:
                labels[neighbour_index] = current_class
                new_neighbours = tree.query_ball_point(X[neighbour_index], r=self.eps)
                if len(new_neighbours) >= self.min_samples:
                    neighbour_indicies.extend(new_neighbours)
            j += 1

    def predict(self):
        """
        Applies the fitted DBScan model to the input data. Splits the training data into clusters.

        Returns:
            labels (torch.Tensor of shape (n_samples,)): The cluster corresponding to each sample. Label -1 indicates, that the algorithm considers that spesific samples as noise.
        Raises:
            NotFittedError: If the DBScan model has not been fitted before predicting.
        """
        if not hasattr(self, "_labels"):
            raise NotFittedError("DBScan.fit() must be called before predicting.")
        
        return self._labels
