import torch
import copy

from ..Kernels import SquaredExponentialCovariance


"""
The support vector machine classifier with the sequential minimal optimization (SMO) quadratic programming solver. The algorithm is based on https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf.
"""
class SVCSMO:
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1, device=torch.device("cpu")):
        self.kernel = kernel
        self.C = torch.tensor(C, device=device)
        self.device = device
    
    def _kernel_matrix(self, X1, X2):
        return self.kernel(X1, X2).to(X1.dtype).to(self.device)

    def fit(self, X, y, epochs=float("inf"), multi_method="ovr", tol=1e-5, epochs_no_change=5):
        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, multi_method, epochs, tol, epochs_no_change)
        self.multiclass = False
        self.transform_y = False
        if set(torch.unique(y)) == {0, 1}:
            y[y == 0] = -1
            self.transform_y = True
        self.y = y.reshape((-1, 1)).to(X.dtype)
        self.X = X
        n = X.shape[0]
        self.K = self._kernel_matrix(X, X)

        self.alpha = torch.zeros(n, 1, dtype=X.dtype, device=self.device)
        self.b = 0

        iter_count = 0
        num_no_change_iter = 0
        while iter_count < epochs:
            num_changed_alphas = 0
            for i in range(n):
                E_i = self._decision_function(self.K[i, :].unsqueeze(-1)) - self.y[i]
                if (self.y[i] * E_i < -tol and self.alpha[i] < self.C) or (self.y[i] * E_i > tol and self.alpha[i] > 0):
                    # Select j randomly
                    j = self._select_j(i, n)
                    E_j = self._decision_function(self.K[j, :].unsqueeze(-1)) - self.y[j]

                    # Compute bounds for α_j
                    if self.y[i] == self.y[j]:
                        L = torch.max(torch.zeros(1, device=self.device), self.alpha[j] + self.alpha[i] - self.C)
                        H = torch.min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = torch.max(torch.zeros(1, device=self.device), self.alpha[j] - self.alpha[i])
                        H = torch.min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    if L == H:
                        continue

                    # Compute η (second derivative of the objective function w.r.t. α_j)
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    # Update α_j
                    alpha_j_old = self.alpha[j].clone()
                    self.alpha[j] = self.alpha[j] - (self.y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = torch.clamp(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < tol:
                        continue

                    # Update α_i
                    alpha_i_old = self.alpha[i].clone()
                    self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # Update bias term
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                num_no_change_iter += 1
            else:
                num_no_change_iter = 0
            if num_no_change_iter >= epochs_no_change:
                break
            iter_count += 1

        self.is_sv = (self.alpha > tol).squeeze()

    def _select_j(self, i, n):
        j = i
        while j == i:
            j = torch.randint(0, n, (1,)).item()
        return j

    def _decision_function(self, K):
        return (self.alpha * self.y).T @ K + self.b

    def _multi_fit(self, X, y, method, epochs, tol, epochs_no_change):
        assert hasattr(self, "multiclass"), "Use SVC.fit(X, y) rather than SVC._multi_fit(X, y) even if your data is multidimensional."
        self.method = method
        if method == "ovr":
            self.n_classes = len(torch.unique(y))
            self.classes = torch.unique(y)
            self.classifiers = []
            for label in self.classes:
                Xs, ys = X, copy.deepcopy(y)
                ys[ys != label], ys[ys == label] = -1, +1
                classifier = SVCSMO(kernel=self.kernel, C=self.C.item(), device=self.device)
                classifier.fit(Xs, ys, epochs=epochs, tol=tol, epochs_no_change=epochs_no_change)
                self.classifiers.append(classifier)

        elif method == "ovo":
            self.n_classes = len(torch.unique(y))
            self.classes = torch.unique(y)
            self.classifiers = []
            for i in range(self.n_classes):
                for j in range(i+1, self.n_classes):
                    class_i, class_j = self.classes[i], self.classes[j]
                    indices = (y == class_i) | (y == class_j)
                    X_subset, y_subset = X[indices], y[indices]
                    y_subset = torch.clone(y_subset)
                    y_subset[y_subset == class_j] = -1
                    y_subset[y_subset == class_i] = +1
                    classifier = SVCSMO(kernel=self.kernel, C=self.C.item())
                    classifier.fit(X_subset, y_subset)
                    self.classifiers.append((classifier, class_i, class_j))
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')

    def predict(self, X, _from_multi=False):
        assert hasattr(self, "multiclass"), "SVC.fit(X, y) must be called before predicting."
        if self.multiclass:
            self.flag = None
            return self._multi_predict(X)
        score = self._decision_function(self._kernel_matrix(self.X, X)).squeeze()
        if _from_multi:
            return score
        return ((torch.sign(score) + 1) / 2).to(torch.int32) if self.transform_y else torch.sign(score).to(torch.int32)

    def _multi_predict(self, X):
        assert hasattr(self, "flag"), "Use SVC.predict(X, y) rather than SVC._multi_predict(X, y) even if your data is multidimensional."
        if self.method == "ovr":
            predictions = torch.zeros((X.shape[0], self.n_classes), device=self.device)
            for i, classifier in enumerate(self.classifiers):
                predictions[:, i] = classifier.predict(X, _from_multi=True)
            return self.classes[torch.argmax(predictions, dim=1)]

        elif self.method == "ovo":
            votes = torch.zeros((X.shape[0], self.n_classes))
            for classifier, class_i, class_j in self.classifiers:
                predictions = classifier.predict(X)
                votes[:, class_i] += (predictions == +1)  # If prediction is +1, vote for class_i
                votes[:, class_j] += (predictions == -1)  # If prediction is -1, vote for class_j
            return self.classes[torch.argmax(votes, dim=1)]
        else:
            raise NotImplementedError('Only "ovr" and "ovo" methods exist for non-binary classification.')
