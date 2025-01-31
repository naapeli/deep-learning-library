import torch
from src.DLL.Data.Preprocessing import OneHotEncoder


X = torch.randint(0, 5, size=(1000, 5))
X[0, 0] = 5
print(X.shape)
encoder = OneHotEncoder()
X_encoded = encoder.fit_encode(X)
print(X_encoded.shape)
result = encoder.decode(X_encoded)
print(result.shape)
print(f"Original data and decoded data are same: {torch.all(X == result).item()}")
