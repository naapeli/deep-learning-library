"""
Character-level language model with attention
================================================

This script trains a small language model with
:class:`~DLL.DeepLearning.Layers.MultiHeadAttention`. The model reads 24 characters at a time and predicts the
next character at every position, like a (very) small GPT. A causal mask makes sure that each position only
attends to itself and the characters before it.

The text is the descriptions of the toy datasets that are shipped with scikit-learn, so nothing has to be
downloaded. The model is compared with the same network without the attention layer, which can only look at
the current character.
"""
import random

import torch
import matplotlib.pyplot as plt
from sklearn import datasets

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import MultiHeadAttention, Dense, Add, Identity
from DLL.DeepLearning.Layers.Regularisation import LayerNorm
from DLL.DeepLearning.Layers.Activations import ReLU, SoftMax
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import CCE
from DLL.Data.Metrics import accuracy


torch.manual_seed(0)
random.seed(0)

# %%
# Data
# ----
#
# The text consists of about 15 000 characters. Its paragraphs are divided randomly into a training set (80 %),
# a validation set (10 %) and a test set (10 %), so the model is evaluated on text it has never seen.
#
# Every character is a token. A sample is a window of 24 consecutive characters, and the target is the same
# window shifted by one character. Each token is encoded as a one-hot vector of the character, followed by a
# one-hot vector of its position in the window. Attention itself does not know the order of the tokens, so the
# position has to be part of the input. To keep the example fast, a new window starts every 5 characters.

loaders = [datasets.load_iris, datasets.load_wine, datasets.load_breast_cancer, datasets.load_diabetes, datasets.load_digits, datasets.load_linnerud]
text = "\n\n".join(load().DESCR for load in loaders)
paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
random.shuffle(paragraphs)
n_train, n_val = int(0.8 * len(paragraphs)), int(0.1 * len(paragraphs))
train_text = "\n\n".join(paragraphs[:n_train])
val_text = "\n\n".join(paragraphs[n_train:n_train + n_val])
test_text = "\n\n".join(paragraphs[n_train + n_val:])

chars = sorted(set(text))
char_to_index = {char: i for i, char in enumerate(chars)}
vocab_size = len(chars)
seq_len = 24


def encode(ids):
    one_hot_chars = torch.eye(vocab_size)[ids]
    positions = torch.eye(seq_len).expand(len(ids), seq_len, seq_len)
    return torch.cat([one_hot_chars, positions], dim=2)


def make_windows(part, stride=5):
    ids = torch.tensor([char_to_index[char] for char in part])
    starts = torch.arange(0, len(ids) - seq_len, stride)
    inputs = torch.stack([ids[start:start + seq_len] for start in starts])
    targets = torch.stack([ids[start + 1:start + seq_len + 1] for start in starts])
    return encode(inputs), torch.eye(vocab_size)[targets]


X_train, Y_train = make_windows(train_text)
X_val, Y_val = make_windows(val_text)
X_test, Y_test = make_windows(test_text)
print(f"{len(text)} characters, {vocab_size} different characters")
print(f"Windows: {len(X_train)} for training, {len(X_val)} for validation and {len(X_test)} for testing")

# %%
# The models
# ----------
#
# The language model embeds every token into 48 features with a dense layer. Then follows a block of masked
# multi-head attention with 4 heads and a residual connection, and a residual feed-forward block, both followed
# by layer normalisation. The last dense layer predicts a probability distribution over the next character at
# every position.
#
# The baseline is the same network without the attention block. Every position is then processed on its own,
# so the baseline can only learn which character usually follows the current one.

embedding_dim = 48


def build_model(use_attention):
    model = Model((seq_len, vocab_size + seq_len))
    model.add(Dense((seq_len, embedding_dim)))
    if use_attention:
        model.add(Add(attention, Identity()))
        model.add(LayerNorm())
    model.add(Add(Dense((seq_len, embedding_dim), activation=ReLU()), Identity()))
    model.add(LayerNorm())
    model.add(Dense((seq_len, vocab_size), activation=SoftMax()))
    model.compile(optimiser=ADAM(learning_rate=0.003), loss=CCE(), metrics=["loss", "accuracy", "val_loss", "val_accuracy"])
    return model


attention = MultiHeadAttention((seq_len, embedding_dim), n_heads=4, use_mask=True)
model = build_model(use_attention=True)
baseline = build_model(use_attention=False)
model.summary()

# %%
# Training
# --------
#
# Both models are trained for 40 epochs. The loss is the cross entropy of the next character, averaged over all
# positions, and the accuracy is the fraction of positions where the most likely character is the correct one.
# The validation set is evaluated after every epoch.

history = model.fit(X_train, Y_train, val_data=(X_val, Y_val), epochs=40, batch_size=64)
history_baseline = baseline.fit(X_train, Y_train, val_data=(X_val, Y_val), epochs=40, batch_size=64)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for hist, name, color in ((history, "with attention", "C0"), (history_baseline, "without attention", "C1")):
    ax[0].plot(hist["loss"], color=color, linestyle="--", label=f"{name}, training")
    ax[0].plot(hist["val_loss"], color=color, label=f"{name}, validation")
    ax[1].plot(hist["accuracy"], color=color, linestyle="--", label=f"{name}, training")
    ax[1].plot(hist["val_accuracy"], color=color, label=f"{name}, validation")
ax[0].set_ylabel("Cross entropy per character")
ax[1].set_ylabel("Next character accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()

# %%
# Test set
# --------
#
# The final models are evaluated on the test set, which was not used during training.

for name, trained_model in (("With attention", model), ("Without attention", baseline)):
    predictions = trained_model.predict(X_test)
    test_loss = CCE().loss(predictions, Y_test).item()
    test_accuracy = accuracy(predictions.reshape(-1, vocab_size), Y_test.reshape(-1, vocab_size))
    print(f"{name}: test loss {test_loss:.3f}, test accuracy {test_accuracy:.3f}")

# %%
# Generating text
# ---------------
#
# The model generates text one character at a time. The window ends with the latest characters, the predicted
# distribution at its last position is sharpened with a temperature of 0.5, and the next character is sampled
# from it. The descriptions contain tables of numbers, so the generated text contains them too.


def generate(model, prompt, n_chars, temperature=0.5):
    ids = [char_to_index[char] for char in prompt]
    for _ in range(n_chars):
        window = torch.tensor(ids[-seq_len:]).unsqueeze(0)
        probabilities = model.predict(encode(window))[0, -1] ** (1 / temperature)
        ids.append(torch.multinomial(probabilities / probabilities.sum(), 1).item())
    return "".join(chars[i] for i in ids)


prompt = "The dataset contains the "
print("With attention:\n" + generate(model, prompt, 200))
print("\nWithout attention:\n" + generate(baseline, prompt, 200))

# %%
# Attention weights
# -----------------
#
# Every head assigns weights to the current and the previous characters, and the weights in each row sum to one.
# The causal mask keeps everything above the diagonal at zero. The first characters can only attend to a few
# characters, so their weights are large. The colour scale of each head is therefore set by the rows from the
# seventh character on, and larger weights are shown in the darkest colour. Spaces are shown as ␣.

sample = "The number of instances "
model.predict(encode(torch.tensor([char_to_index[char] for char in sample]).unsqueeze(0)))
weights = attention.attention_scores[0]
labels = ["␣" if char == " " else char for char in sample]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for head, ax in enumerate(axes):
    ax.imshow(weights[head], cmap="Blues", vmin=0, vmax=weights[head, 6:].max())
    ax.set_title(f"Head {head + 1}")
    ax.set_xticks(range(seq_len), labels, fontsize=7)
    ax.set_yticks(range(seq_len), labels, fontsize=7)
    ax.set_xlabel("Attended character")
axes[0].set_ylabel("Current character")
plt.tight_layout()
plt.show()
