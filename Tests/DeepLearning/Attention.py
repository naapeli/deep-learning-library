"""
Deep learning with Attention
==============================

This script implements a model to predict a dummy dataset using MultiHeadAttention. The model 
has a similar structure to modern large language models, but with way less parameters.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import MultiHeadAttention, Dense, Flatten, Reshape, LayerList
from DLL.DeepLearning.Layers.Regularisation import LayerNorm
from DLL.DeepLearning.Layers.Activations import ReLU
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import MSE
from DLL.Data.Preprocessing import data_split
from DLL.Data.Metrics import mean_squared_error

# %%
# Data
# ----
#
# Every sample is a sequence of 10 numbers, and the target is the sum of their squares.

n = 1000
seq_len = 10

X = 10 * torch.rand((n, seq_len))
y = (X ** 2).sum(dim=1)

X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y, 0.6, 0.2)

# %%
# The model
# ---------
#
# A dense layer lifts the input to 99 values, which are reshaped into a sequence of 11 tokens with 9 features.
# Three copies of a block of multi-head attention, layer normalisation and a dense layer follow, before the
# output is flattened and mapped to a single number.

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

# %%
# Training
# --------

history = model.fit(X_train, y_train, val_data=(X_val, y_val), epochs=100, callback_frequency=1, batch_size=64, verbose=True)
y_pred = model.predict(X_test)
print(f"Test mean squared error: {mean_squared_error(y_pred, y_test)}")

# %%
# Learning curves
# ---------------

plt.figure(figsize=(8, 8))
plt.semilogy(history["loss"], label="loss")
plt.semilogy(history["val_loss"], label="validation loss")
plt.legend()
plt.show()
