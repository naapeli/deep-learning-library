import torch


def _nearest_centroid(X, centroids, chunk_size=8192):
    """
    Assigns every row of X to its nearest centroid.

    The squared distances are computed as ||x||^2 - 2 x.c + ||c||^2 in chunks of rows, so the
    memory use is O(chunk_size * k) instead of the O(n * k * d) of a broadcasted difference.

    Returns:
        labels (torch.Tensor of shape (n_samples,)): Index of the nearest centroid.
        squared_distances (torch.Tensor of shape (n_samples,)): Squared distance to that centroid.
    """
    centroids = centroids.to(dtype=X.dtype, device=X.device)
    centroid_sq = (centroids * centroids).sum(dim=1)
    labels = torch.empty(len(X), dtype=torch.long, device=X.device)
    squared_distances = torch.empty(len(X), dtype=X.dtype, device=X.device)
    for start in range(0, len(X), chunk_size):
        x = X[start:start + chunk_size]
        # ||x||^2 does not change the argmin, so it is only added for the returned distances.
        scores = centroid_sq[None, :] - 2.0 * (x @ centroids.T)
        min_scores, chunk_labels = scores.min(dim=1)
        labels[start:start + chunk_size] = chunk_labels
        squared_distances[start:start + chunk_size] = ((x * x).sum(dim=1) + min_scores).clamp_min(0.0)
    return labels, squared_distances


def _cluster_sums(X, labels, k, dtype=None):
    """Per-cluster counts and sums of the rows of X."""
    dtype = X.dtype if dtype is None else dtype
    counts = torch.bincount(labels, minlength=k).to(dtype)
    sums = torch.zeros((k, X.shape[1]), dtype=dtype, device=X.device).index_add_(0, labels, X.to(dtype))
    return counts, sums
