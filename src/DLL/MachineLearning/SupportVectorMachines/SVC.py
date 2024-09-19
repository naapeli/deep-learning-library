import torch

from ..Kernels import SquaredExponentialCovariance
from ...DeepLearning.Optimisers.ADAM import Adam


class SVC:
    def __init__(self, kernel=SquaredExponentialCovariance(), C=1, learning_rate=0.001, device=torch.device("cpu")):
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate
        self.device = device
    
    def _kernel_matrix(self, X1, X2):
        covariance = torch.tensor([[self.kernel(x1, x2) for x1 in X1] for x2 in X2], dtype=X1.dtype, device=self.device).T
        return covariance
    
    def fit(self, X, y, epochs=100, optimiser=None):
        if len(torch.unique(y)) > 2:
            self.multiclass = True
            return self._multi_fit(X, y, epochs=epochs, optimiser=optimiser)
        self.multiclass = False
        if set(torch.unique(y)) == {0, 1}: y[y == 0] = -1
        self.y = torch.where(y <= 0, -1, 1).to(X.dtype)
        self.transform_y = not torch.all(self.y == y)
        self.X = X

        self.weight = torch.randn(X.shape[1], dtype=X.dtype)
        self.bias = torch.zeros(1, dtype=X.dtype)

        optimiser = Adam(self.learning_rate) if optimiser is None else optimiser
        optimiser.initialise_parameters([self.weight, self.bias])
        # self.K = self._kernel_matrix(X, X)

        for epoch in range(epochs):
            hinge_loss = 0
            for x_i, y_i in zip(self.X, self.y):
                # gradient of the hinge loss with l2 regularisation
                comparison = (1 - y_i * (x_i @ self.weight - self.bias)).item()
                if 0 >= comparison:
                    self.weight.grad = self.learning_rate * (2 * self.weight)
                    self.bias.grad = torch.zeros_like(self.bias)
                else:
                    hinge_loss += comparison
                    self.weight.grad = self.learning_rate * (2 * self.weight - self.C * x_i * y_i)
                    self.bias.grad = torch.tensor([self.learning_rate * y_i])
            
                # update the parameters
                optimiser.update_parameters()
            hinge_loss *= self.C / self.X.shape[1]
            hinge_loss += (self.weight ** 2).sum().item()
            print(f"Epoch: {epoch + 1} - Hinge loss: {round(hinge_loss, 2)}")

    def _multi_fit(X, y, epochs=100):
        pass

    def predict(self, X):
        assert hasattr(self, "multiclass"), "SVC.fit(X, y) must be called before predicting."
        if self.multiclass: return self._multi_predict(X)
        prediction = X @ self.weight - self.bias
        return ((torch.sign(prediction) + 1) / 2).to(torch.int32) if self.transform_y else torch.sign(prediction).to(torch.int32)

    def _multi_predict(self, X):
        raise NotImplementedError()
