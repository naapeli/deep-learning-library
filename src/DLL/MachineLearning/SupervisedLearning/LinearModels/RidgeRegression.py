import torch


class RidgeRegression:
    def fit(self, X, Y):
        if len(X.shape) == 1: X = X.unsqueeze(1)
        self.X = X
        self.Y = Y
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        self.beta = torch.linalg.pinv(X.T @ X) @ X.T @ Y
        self.residuals = self.Y - self.predict(self.X)

    def predict(self, X):
        assert hasattr(self, "beta"), "RidgeRegression.fit(x, y) must be called before predicting"
        if len(X.shape) == 1: X = X.unsqueeze(1)
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        return X @ self.beta
