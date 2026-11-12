import math

import torch

from ._centroid_utils import _nearest_centroid, _cluster_sums
from ....Exceptions import NotFittedError


# Constants from the paper (Bessa et al., 2026).
_COSINE_FLOOR = 0.1       # the last step of the cosine schedule is 0.1 * alpha0
_REVIVE_EVERY = 5         # dead clusters are checked every 5 batches
_REVIVE_TAU = 0.1         # a cluster is dead if its mass is below tau * mean mass
_SPLIT_DELTA = 2.0 ** -7  # relative perturbation used when splitting a donor cluster
_INIT_MASS = 1e-10        # the initial centroids get a negligible weight
# Step law: alpha0 = min(c * k^a_k * B^a_B * n^a_n * T^a_T, 1)
_STEP_LAW = dict(c=0.824, k=-0.46, B=0.90, n=-0.45, T=-0.11)
# Batch law: B = c * k^b_k * n^b_n * T^b_T, rounded to a power of two and capped at n.
_BATCH_LAW = dict(c=0.035, k=0.13, n=0.88, T=0.81)
# Smaller batches for very large n (as in the paper) and low-dimensional data (as in the official code).
_BATCH_LAW_SMALL_C = 0.0044


def _round_pow2(x):
    if x <= 1:
        return 1
    lower = 2 ** int(math.floor(math.log2(x)))
    return lower if x / lower < (2 * lower) / x else 2 * lower


class EMeansClustering:
    """
    e-means clustering: mini-batch k-means where every centroid is an exponential moving average (EMA) of the points assigned to it.

    Every cluster j keeps an EMA mass N_j and an EMA sum S_j, and the centroid is c_j = S_j / N_j. For each mini-batch, the points are assigned to their nearest centroid and, with the per-batch counts n_j and sums s_j,

    .. math::

        N_j \\leftarrow \\gamma N_j + (1 - \\gamma) n_j, \\quad S_j \\leftarrow \\gamma S_j + (1 - \\gamma) s_j.

    The step :math:`\\alpha = 1 - \\gamma` follows a cosine schedule from :math:`\\alpha_0` down to :math:`0.1\\alpha_0`. The batch size and :math:`\\alpha_0` come from scaling laws fitted in the paper, so there is nothing to tune. Every 5 batches, clusters whose mass dropped below `revival_threshold` times the mean mass are revived by splitting a donor cluster. After the last epoch, every centroid is replaced by the mean of the points assigned to it during that epoch.

    Args:
        k (int, optional): The number of clusters. Defaults to 3. Must be a positive integer.
        n_epochs (int, optional): The number of passes over the data. The paper reports that 2-3 epochs typically beat 25 iterations of Lloyd's algorithm. Defaults to 3. Must be a positive integer.
        batch_size (int | None, optional): The mini-batch size. If None, it is chosen by the batch size scaling law. Defaults to None.
        alpha0 (float | None, optional): The peak EMA step in (0, 1]. If None, it is chosen by the step scaling law. Defaults to None.
        revival_threshold (float | None, optional): The threshold tau for reviving dead clusters. 0 disables revival. If None, 0.1 is used, except 0 for data with at most 2 features, where revival hurts the held-out error. Defaults to None.
    Attributes:
        centroids (torch.Tensor of shape (k, n_features)): The final centroids.
        inertia (float): The total squared distance of the training points to their nearest final centroid.
        labels (torch.Tensor of shape (n_samples,)): The cluster of each training point with respect to the final centroids.
        history (list[float]): The mean squared distance of the points of each epoch to the centroids at the moment they were assigned.
        resolved_batch_size (int): The batch size that was used.
        resolved_alpha0 (float): The peak EMA step that was used.
        n_revived (int): The number of revived clusters.

    Reference:
        S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze, H. Jégou, "e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501, 2026.
    """
    def __init__(self, k=3, n_epochs=3, batch_size=None, alpha0=None, revival_threshold=None):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(n_epochs, int) or n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer.")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be None or a positive integer.")
        if alpha0 is not None and (not isinstance(alpha0, int | float) or not 0 < alpha0 <= 1):
            raise ValueError("alpha0 must be None or a real number in (0, 1].")
        if revival_threshold is not None and (not isinstance(revival_threshold, int | float) or revival_threshold < 0):
            raise ValueError("revival_threshold must be None or a non-negative real number.")
        self.k = k
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.revival_threshold = revival_threshold

    @staticmethod
    def batch_size_law(n, k, n_epochs, n_features=None):
        """
        The batch size given by the scaling law of the paper.

        Args:
            n (int): The number of samples.
            k (int): The number of clusters.
            n_epochs (int): The number of epochs.
            n_features (int | None, optional): The number of features. Data with at most 16 features uses smaller batches. Defaults to None.
        Returns:
            int: The batch size.
        """
        small = n > 2_000_000 or (n_features is not None and n_features <= 16)
        c = _BATCH_LAW_SMALL_C if small else _BATCH_LAW["c"]
        raw = c * k ** _BATCH_LAW["k"] * n ** _BATCH_LAW["n"] * n_epochs ** _BATCH_LAW["T"]
        return int(max(1, min(_round_pow2(raw), n)))

    @staticmethod
    def step_law(n, k, n_epochs, batch_size):
        """
        The peak EMA step given by the scaling law of the paper.

        Args:
            n (int): The number of samples.
            k (int): The number of clusters.
            n_epochs (int): The number of epochs.
            batch_size (int): The batch size.
        Returns:
            float: The peak step alpha0.
        """
        a = (_STEP_LAW["c"] * k ** _STEP_LAW["k"] * batch_size ** _STEP_LAW["B"]
             * n ** _STEP_LAW["n"] * n_epochs ** _STEP_LAW["T"])
        return float(min(max(a, 1e-4), 1.0))

    @staticmethod
    def _cosine_step(alpha0, t, total_steps):
        cos = 0.5 * (1.0 + math.cos(math.pi * t / total_steps))
        return alpha0 * (_COSINE_FLOOR + (1.0 - _COSINE_FLOOR) * cos)

    def fit(self, X):
        """
        Fits the EMeansClustering model to the input data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature. The number of samples must be atleast `k`.
        Returns:
            None
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] < self.k:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast k samples.")
        if not torch.is_floating_point(X):
            X = X.to(torch.get_default_dtype())

        n, d = X.shape
        k, T = self.k, self.n_epochs
        B = min(self.batch_size or self.batch_size_law(n, k, T, d), n)
        alpha0 = float(self.alpha0) if self.alpha0 is not None else self.step_law(n, k, T, B)
        tau = self.revival_threshold if self.revival_threshold is not None else (0.0 if d <= 2 else _REVIVE_TAU)
        total_steps = T * math.ceil(n / B)

        # The state is kept in double precision. Instead of S_j and N_j, the centroid c_j = S_j / N_j and N_j are
        # stored, which is the same algebra but avoids 0 / 0 for clusters whose mass goes to zero.
        acc = torch.float64
        centroids = X[torch.randperm(n, device=X.device)[:k]].to(acc)
        mass = torch.full((k,), _INIT_MASS, dtype=acc, device=X.device)

        self.history = []
        self.n_revived = 0
        t = 0
        for epoch in range(T):
            last_epoch = epoch == T - 1
            if last_epoch:
                epoch_sums = torch.zeros((k, d), dtype=acc, device=X.device)
                epoch_counts = torch.zeros(k, dtype=acc, device=X.device)
            squared_error = 0.0
            order = torch.randperm(n, device=X.device)
            for start in range(0, n, B):
                batch = X[order[start:start + B]]

                # 1. assign the batch to the current centroids
                labels, squared_distances = _nearest_centroid(batch, centroids)
                squared_error += squared_distances.sum().item()
                counts, sums = _cluster_sums(batch, labels, k, dtype=acc)
                present = counts > 0

                # 2. EMA update with the scheduled step
                alpha = self._cosine_step(alpha0, t, total_steps)
                mass *= 1.0 - alpha
                new_mass = mass[present] + alpha * counts[present]
                weight = alpha * counts[present] / new_mass
                batch_means = sums[present] / counts[present, None]
                centroids[present] += weight[:, None] * (batch_means - centroids[present])
                mass[present] = new_mass

                if last_epoch:
                    epoch_sums += sums
                    epoch_counts += counts

                # 3. revive dead clusters every few batches
                t += 1
                if tau > 0 and t % _REVIVE_EVERY == 0:
                    self.n_revived += self._revive(centroids, mass, tau)

            self.history.append(squared_error / n)

        # 4. Every centroid becomes the mean of the points assigned to it during the last epoch.
        nonempty = epoch_counts > 0
        centroids[nonempty] = epoch_sums[nonempty] / epoch_counts[nonempty, None]

        self.centroids = centroids.to(X.dtype)
        self.labels, squared_distances = _nearest_centroid(X, self.centroids)
        self.inertia = squared_distances.sum().item()
        self.resolved_batch_size = B
        self.resolved_alpha0 = alpha0

    def _revive(self, centroids, mass, tau):
        # Dead clusters are re-seeded by splitting donors drawn with probability proportional to their mass.
        threshold = tau * mass.mean()
        dead = torch.nonzero(mass < threshold).flatten()
        candidates = torch.nonzero((mass >= threshold) & (mass > 0)).flatten()
        m = min(len(dead), len(candidates))
        if m == 0:
            return 0
        # Weighted sampling without replacement with the Gumbel top-m trick.
        uniform = torch.rand(len(candidates), dtype=mass.dtype, device=mass.device).clamp(1e-12, 1 - 1e-12)
        keys = torch.log(mass[candidates]) - torch.log(-torch.log(uniform))
        donors = candidates[torch.topk(keys, m).indices]
        targets = dead[:m]

        # child = c * [1 + delta, 1 - delta, ...] and donor = c * [1 - delta, 1 + delta, ...]
        sign = torch.ones(centroids.shape[1], dtype=centroids.dtype, device=centroids.device)
        sign[1::2] = -1.0
        base = centroids[donors]
        child = base * (1.0 + _SPLIT_DELTA * sign)
        donor = base * (1.0 - _SPLIT_DELTA * sign)
        # An all-zero donor would give two identical centroids, so one is nudged.
        same = (child == donor).all(dim=1)
        child[same, 0] += _SPLIT_DELTA
        centroids[targets] = child
        centroids[donors] = donor

        # Both halves get half of the donor's mass.
        mass[donors] *= 0.5
        mass[targets] = mass[donors]
        return m

    def predict(self, X):
        """
        Applies the fitted EMeansClustering model to the input data, partitioning it to `k` clusters.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be partitioned.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The cluster corresponding to each sample.
        Raises:
            NotFittedError: If the EMeansClustering model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "centroids"):
            raise NotFittedError("EMeansClustering.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.centroids.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return _nearest_centroid(X.to(self.centroids.dtype), self.centroids)[0]
