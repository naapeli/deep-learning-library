import torch
import matplotlib.pyplot as plt

from src.DLL.DeepLearning.Model import Model
from src.DLL.DeepLearning.Layers import MultiHeadAttention, Dense, Flatten, Reshape, LayerList
from src.DLL.DeepLearning.Layers.Regularisation import LayerNorm
from src.DLL.DeepLearning.Layers.Activations import ReLU
from src.DLL.DeepLearning.Optimisers import ADAM
from src.DLL.DeepLearning.Losses import MSE
from src.DLL.Data.Preprocessing import data_split
from src.DLL.Data.Metrics import mean_squared_error


n = 1000
seq_len = 10

X = 10 * torch.rand((n, seq_len))
y = (X ** 2).sum(dim=1)

X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y, 0.6, 0.2)

block = LayerList(
    MultiHeadAttention((11, 9), n_heads=3, dropout=0.5),
    LayerNorm(),
    Dense((11, 9)),
    ReLU()
)

model = Model((seq_len,))
model.add(Dense(99, activation=ReLU()))
model.add(Reshape((11, 9)))
model.add(block.clone())
model.add(block.clone())
model.add(block.clone())
model.add(Flatten())
model.add(Dense(tuple()))
model.compile(ADAM(), MSE(), metrics=["loss", "val_loss"])
model.summary()

history = model.fit(X_train, y_train, val_data=(X_val, y_val), epochs=1000, callback_frequency=10, batch_size=64, verbose=True)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
print(f"Test mean squared error: {mean_squared_error(y_pred, y_test)}")
rel_tol = 0.1
# print(torch.isclose(y_pred, y_test, rtol=rel_tol))
print(torch.mean(torch.isclose(y_pred, y_test, rtol=rel_tol).float()).item())

plt.semilogy(history["loss"], label="loss")
plt.semilogy(history["val_loss"], label="validation loss")
plt.legend()
plt.show()
